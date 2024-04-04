import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import soundfile as sf
import random
import numpy as np
import cv2
from moviepy.editor import *
from typing import Tuple
from utils.utils_whisper import log_mel_spectrogram_whisper
sys.stdout.flush()

EPS = np.finfo("float").eps


class dataset_LRS3(torch.utils.data.Dataset):
    """
        Data provider for the LRS3 dataset including audio and video.
    """

    def __init__(self, 
                 lrs3_prep_dir: str,
                 lrs3_lists_path: str,
                 lrs3_filelist: str, 
                 lrs3_audio_dir: str,
                 lrs3_video_dir: str,
                 musan_prep_dir: str,
                 musan_type: str,       # Options: all, babble, music, noise, lrs3_sidespeaker
                 mode: str,             # Options: train, valid, test
                 specaugment_dict: dict,
                 SNR_test: float = 0,
                 SNR_range_train: Tuple[int, int] = (-20, 50),
                 SNR_range_valid: Tuple[int, int] = (-10, 10),
                 logfile_fn: str = None,
                 set_seed: bool = True,
                 seed: int = 1,
                 maxLenVideo: int = 30*25,
                 frameCropShape: Tuple[int, int] = (88, 88),
                 flipProb: float = 0.5,
                 erasingProb: float = 0.5,
                 eraseSizeRange: Tuple[int, int] = (2, 30),
                 eraseRatioRange: Tuple[float, float] = (0.3, 3.3),
                 timeMaskRange: Tuple[int, int] = (0, 10)):
        
        """
        Initialize the dataset with specified parameters.

        Args:
        - lrs3_prep_dir (str): Directory containing LRS3 preparation data.
        - lrs3_lists_path (str): Path to lists of files.
        - lrs3_filelist (str): Name of the file list.
        - lrs3_audio_dir (str): Directory containing LRS3 audio data.
        - lrs3_video_dir (str): Directory containing LRS3 video data.
        - musan_prep_dir (str): Directory containing MUSAN preparation data.
        - musan_type (str): Type of MUSAN data to use. Options: all, babble, music, noise.
        - mode (str): Mode of operation. Options: train, valid, test.
        - specaugment_dict (dict): Definitions for spec augmentation
        - SNR_test (float): Signal-to-Noise Ratio for test mode.
        - SNR_range_train (Tuple[int, int]): SNR range for training mode.
        - SNR_range_valid (Tuple[int, int]): SNR range for validation mode.
        - logfile_fn: File to log output to.
        - set_seed (bool): Whether to set seed.
        - seed (int): Defined seed.
        - maxLenVideo (int): Maximum video length in frames.
        - frameCropShape (Tuple[int, int]): Shape of cropped frames.
        - flipProb (float): Probability of frame flipping.
        - erasingProb (float): Probability of erasing.
        - eraseSizeRange (Tuple[int, int]): Range of erase size.
        - eraseRatioRange (Tuple[float, float]): Range of erase ratio.
        - timeMaskRange (Tuple[int, int]): Range of time mask.
        """

        self.lrs3_prep_dir = lrs3_prep_dir
        self.lrs3_lists_path = lrs3_lists_path
        self.lrs3_filelist = lrs3_filelist
        self.lrs3_audio_dir = lrs3_audio_dir
        self.lrs3_video_dir = lrs3_video_dir
        self.musan_prep_dir = musan_prep_dir
        self.musan_type = musan_type
        self.mode = mode
        self.specaugment_dict = specaugment_dict
        self.SNR_test = SNR_test
        self.SNR_range_train = SNR_range_train
        self.SNR_range_valid = SNR_range_valid
        self.logfile_fn = logfile_fn
        self.set_seed = set_seed
        self.seed = seed
        self.maxLenVideo = maxLenVideo

        # Augmentation settings
        self.frameCropShape = frameCropShape
        self.flipProb = flipProb
        self.erasingProb = erasingProb
        self.eraseSizeRange = eraseSizeRange
        self.eraseRatioRange = eraseRatioRange 
        self.timeMaskRange = timeMaskRange

        # file and label.list for set
        self.filelist_fname = os.path.join(self.lrs3_lists_path, f'file.list.{self.lrs3_filelist}')
        self.labellist_fname = os.path.join(self.lrs3_lists_path, f'label.list.{self.lrs3_filelist}')

        # Prepare file and label lists
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as logfile:
                print("Prepare dataset ...", file=logfile)
        self.fids = list()
        for el in open(self.filelist_fname, "r").readlines()[:]:
            el = el.strip('\n')
            self.fids.append(el)
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as logfile:
                print(f'file_list fids - Length: {len(self.fids)}', file=logfile)

        self.label_list_lrs3 = list()
        for el in open(self.labellist_fname, "r").readlines()[:]:
            el = el.strip('\n')
            self.label_list_lrs3.append(el)
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as logfile:
                print(f'label_list label_list_lrs3 - Length: {len(self.label_list_lrs3)}', file=logfile)

        # Musan categories - all, babble, music, noise
        self.filelist_musan_fname_all = os.path.join(self.musan_prep_dir, 'tsv', 'all', f'{mode}.tsv')
        self.file_list_musan_all = list()
        for el in open(self.filelist_musan_fname_all, "r").readlines()[:]:
            el = el.strip('\n')
            self.file_list_musan_all.append(el)
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as logfile:
                print(f'file_list file_list_musan_all - Length: {len(self.file_list_musan_all)}', file=logfile)

        self.filelist_musan_fname_babble = os.path.join(self.musan_prep_dir, 'tsv', 'babble', f'{mode}.tsv')
        self.file_list_musan_babble = list()
        for el in open(self.filelist_musan_fname_babble, "r").readlines()[:]:
            el = el.strip('\n')
            self.file_list_musan_babble.append(el)
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as logfile:
                print(f'file_list file_list_musan_babble - Length: {len(self.file_list_musan_babble)}', file=logfile)

        self.filelist_musan_fname_music = os.path.join(self.musan_prep_dir, 'tsv', 'music', f'{mode}.tsv')
        self.file_list_musan_music = list()
        for el in open(self.filelist_musan_fname_music, "r").readlines()[:]:
            el = el.strip('\n')
            self.file_list_musan_music.append(el)
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as logfile:
                print(f'file_list file_list_musan_music - Length: {len(self.file_list_musan_music)}', file=logfile)

        self.filelist_musan_fname_noise = os.path.join(self.musan_prep_dir, 'tsv', 'noise', f'{mode}.tsv')
        self.file_list_musan_noise = list()
        for el in open(self.filelist_musan_fname_noise, "r").readlines()[:]:
            el = el.strip('\n')
            self.file_list_musan_noise.append(el)
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as logfile:
                print(f'file_list file_list_musan_noise - Length: {len(self.file_list_musan_noise)}', file=logfile)

        self.fids_lrs3_sidespeakers = self.fids
        self.file_list_lrs3_sidespeaker = list()
        for fid in self.fids_lrs3_sidespeakers:
            self.file_list_lrs3_sidespeaker.append(os.path.join(self.lrs3_audio_dir, f'{fid}.wav'))
        if self.logfile_fn is not None:
            with open(self.logfile_fn, 'a') as logfile:
                print(f'file_list file_list_lrs3_sidespeaker - Length: {len(self.file_list_lrs3_sidespeaker)}', file=logfile)

        # Set seed for torch
        if self.set_seed:
            if self.logfile_fn is not None:
                with open(self.logfile_fn, 'a') as logfile:
                    print(f"Set torch.manual_seed {self.seed}", file=logfile)
            torch.manual_seed(self.seed)

        self.musan_all_dedicated_idx    = torch.randint(0, len(self.file_list_musan_all), (len(self.fids),))
        self.musan_babble_dedicated_idx = torch.randint(0, len(self.file_list_musan_babble), (len(self.fids),))
        self.musan_music_dedicated_idx  = torch.randint(0, len(self.file_list_musan_music), (len(self.fids),))
        self.musan_noise_dedicated_idx  = torch.randint(0, len(self.file_list_musan_noise), (len(self.fids),))
        self.lrs3_sidespeaker_dedicated_idx  = torch.randint(0, len(self.file_list_lrs3_sidespeaker), (len(self.fids),))

        # Set SNR values for val mode
        self.SNR_val_dedicated_valid = np.array(torch.rand(len(self.fids)) * (self.SNR_range_valid[1]-self.SNR_range_valid[0]) + self.SNR_range_valid[0])



    def __len__(self):
        'Denotes the total number of samples'
        return len(self.fids)
    

    def prepare_audio_snr__(self, signal: np.ndarray, noise: np.ndarray, snr_value: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare audio data with a given SNR value.

        Args:
        - signal (np.ndarray): The signal (audio data).
        - noise (np.ndarray): The noise data.
        - snr_value (float): Desired SNR value.

        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing processed audio data, original signal, and original noise.
        """
        # Combine audio data
        if len(signal)>len(noise):
            factor = int(len(signal)/len(noise))+1
            noise_list = [noise for i in range(factor)]
            noise = np.concatenate(noise_list)
        
        if self.mode=='train':
            if len(noise)-len(signal)==0:
                start_id = 0
            else:
                start_id = torch.randint(0, len(noise)-len(signal), (1,))
        else:
            start_id = 0
        noise_part = noise[start_id:start_id+len(signal)]
        noise = noise_part

        signal_sil = self.remove_silent_frames(x=signal, dyn_range=30)
        noise_sil = self.remove_silent_frames(x=noise, dyn_range=30)

        p_signal_sil = np.sum(signal_sil**2)/len(signal_sil)
        p_noise_sil = np.sum(noise_sil**2)/len(noise_sil)

        targ_noise_p = p_signal_sil/(10**(snr_value/10))
        noise_factor = (targ_noise_p/p_noise_sil)**0.5
        noise = noise*noise_factor.item()

        print(signal)
        print(noise)
        print(noise_factor)
        print(jksa)

        signal_sil = self.remove_silent_frames(x=signal, dyn_range=30)
        noise_sil = self.remove_silent_frames(x=noise, dyn_range=30)

        p_signal_sil = np.sum(signal_sil**2)/len(signal_sil)
        p_noise_sil = np.sum(noise_sil**2)/len(noise_sil)

        SNR_val = 10 * (np.log10(p_signal_sil) - np.log10(p_noise_sil))

        
        # print(SNR_val)
        # print(f'lrs3_dat: len {len(signal)} - p {p_signal}')
        # print(f'noise: len {len(noise)} - p {p_noise}')
        # print(f'SNR: {SNR_val}')

        return signal+noise, signal, noise


    def prepare_audio_snr(self, signal: np.ndarray, noise: np.ndarray, snr_value: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare audio data with a given SNR value.

        Args:
        - signal (np.ndarray): The signal (audio data).
        - noise (np.ndarray): The noise data.
        - snr_value (float): Desired SNR value.

        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing processed audio data, original signal, and original noise.
        """
        # Combine audio data
        if len(signal)>len(noise):
            factor = int(len(signal)/len(noise))+1
            noise_list = [noise for i in range(factor)]
            noise = np.concatenate(noise_list)
        
        if self.mode=='train':
            if len(noise)-len(signal)==0:
                start_id = 0
            else:
                start_id = np.random.randint(0, len(noise)-len(signal), 1).flatten()[0]
        else:
            start_id = 0
        noise_part = noise[start_id:start_id+len(signal)]
        noise = noise_part

        signal_sil = self.remove_silent_frames(x=signal, dyn_range=30)
        noise_sil = self.remove_silent_frames(x=noise, dyn_range=30)

        p_signal_sil = np.sum(signal_sil**2)/len(signal_sil)
        p_noise_sil = np.sum(noise_sil**2)/len(noise_sil)

        targ_noise_p = p_signal_sil/(10**(snr_value/10))
        noise_factor = (targ_noise_p/p_noise_sil)**0.5
        noise = noise*noise_factor
        
        signal_sil = self.remove_silent_frames(x=signal, dyn_range=30)
        noise_sil = self.remove_silent_frames(x=noise, dyn_range=30)

        p_signal_sil = np.sum(signal_sil**2)/len(signal_sil)
        p_noise_sil = np.sum(noise_sil**2)/len(noise_sil)

        SNR_val = 10 * (np.log10(p_signal_sil) - np.log10(p_noise_sil))

        return signal+noise, signal, noise


    def centreCrop_videoframes(self, frames: np.ndarray) -> np.ndarray:
        """
        Center-crop video frames.

        Args:
        - frames (np.ndarray): Array of video frames.

        Returns:
        - np.ndarray: Cropped video frames.
        """
        # Crop
        start_H = int((frames.shape[2]-self.frameCropShape[0])/2)
        start_W = int((frames.shape[3]-self.frameCropShape[1])/2)
        frames = frames[:,:, start_H:start_H+self.frameCropShape[0], start_W:start_W+self.frameCropShape[1]]
        return frames
    
    
    def augment_videoframes(self, frames: np.ndarray) -> np.ndarray:
        """
        Augment video frames with flipping, erasing, and time masking.

        Args:
        - frames (np.ndarray): Array of video frames with shape (1, SequL, H, W).

        Returns:
        - np.ndarray: Augmented video frames.
        """
        frames_meanValue = np.mean(frames)

        # Crop
        start_H = torch.randint(0, frames.shape[2]-self.frameCropShape[0], (1,))
        start_W = torch.randint(0, frames.shape[3]-self.frameCropShape[1], (1,))
        frames = frames[:,:, start_H:start_H+self.frameCropShape[0], start_W:start_W+self.frameCropShape[1]]

        # Flip
        if torch.rand(1)<self.flipProb:
            frames = np.flip(frames, 3)
        
        # Erasing
        for i in range(frames.shape[1]):
            if torch.rand(1)<self.erasingProb:
                er_S = torch.randint(self.eraseSizeRange[0], self.eraseSizeRange[1], (1,))
                er_R = torch.rand(1)*(self.eraseRatioRange[1]-self.eraseRatioRange[0]) + self.eraseRatioRange[0]
                if er_R>1.0:
                    er_H = er_S.item()
                    er_W = int(np.round(er_S/er_R))
                else:
                    er_W = er_S.item()
                    er_H = int(np.round(er_S*er_R))
                
                # Mask Frame
                start_H = torch.randint(0, frames.shape[2]-er_H, (1,))
                start_W = torch.randint(0, frames.shape[3]-er_W, (1,))
                frames[0,i,start_H:start_H+er_H, start_W:start_W+er_W] = frames_meanValue
        
        # Time masking
        sec_l = int(frames.shape[1]/25)
        for i in range(sec_l):
            mask_l = torch.randint(self.timeMaskRange[0], self.timeMaskRange[1], (1,)).item()
            mask_start = torch.randint(0, 25-mask_l, (1,)) + i*25
            frames[:,mask_start:mask_start+mask_l,:,:] = frames_meanValue

        return frames


    # Spec-augmentation
    # roughly based on https://arxiv.org/pdf/1904.08779.pdf
    def augment_spec(self, melspectrum: torch.Tensor, specaugment_dict: dict) -> torch.Tensor:
        """
        Apply SpecAugment augmentation to the mel-spectrogram.

        Args:
        - melspectrum (torch.Tensor): Mel-spectrogram tensor.
        - specaugment_dict (dict): Dictionary containing SpecAugment configuration.

        Returns:
        - torch.Tensor: Augmented mel-spectrogram tensor.
        """
        melspectrum_aug = melspectrum.clone()
        spec_length = melspectrum_aug.shape[0]
        max_specauglength = min([specaugment_dict['sequMaxLength'], int(specaugment_dict['maxSpecLenRatio']*spec_length)])
        aug_length = torch.randint(0, max_specauglength, (1,)).item()
        if aug_length>0:
            spec_start = torch.randint(0, spec_length-aug_length, (1,)).item()
            melspectrum_aug[spec_start:spec_start+aug_length,:] = 0
        aug_channels = torch.randint(0, specaugment_dict['maxChannels'], (1,)).item()
        if aug_channels>0:
            spec_start = torch.randint(0, melspectrum_aug.shape[-1]-aug_channels, (1,)).item()
            melspectrum_aug[:, spec_start:spec_start+aug_channels] = 0
        return melspectrum_aug


    def __getitem__(self, index):

        # Get noise category
        if self.mode=='train':
            # Get random musan category
            musan_cat = random.choice(['babble', 'music', 'noise', 'lrs3_sidespeaker'])
        else:
            musan_cat = self.musan_type
        
        if musan_cat=='babble':
            musan_fn_list = self.file_list_musan_babble
            musan_dedicated_idx = self.musan_babble_dedicated_idx 
        elif musan_cat=='music':
            musan_fn_list = self.file_list_musan_music
            musan_dedicated_idx = self.musan_music_dedicated_idx 
        elif musan_cat=='noise':
            musan_fn_list = self.file_list_musan_noise
            musan_dedicated_idx = self.musan_noise_dedicated_idx 
        elif musan_cat=='lrs3_sidespeaker':
            musan_fn_list = self.file_list_lrs3_sidespeaker
            musan_dedicated_idx = self.lrs3_sidespeaker_dedicated_idx 
        elif musan_cat=='all':
            musan_fn_list = self.file_list_musan_all
            musan_dedicated_idx = self.musan_all_dedicated_idx 
        
        # Get noise index
        if self.mode=='train':
            musan_idx_select = torch.randint(0,len(musan_fn_list), (1,))
        else:
            musan_idx_select = musan_dedicated_idx[index]
        
        # Get musan fname
        noise_fname = musan_fn_list[musan_idx_select]

        # filename and dat_label
        fid = self.fids[index]
        fn_audio = os.path.join(self.lrs3_audio_dir, f'{fid}.wav')
        fn_video = os.path.join(self.lrs3_video_dir, f'{fid}.mp4')
        dat_label = self.label_list_lrs3[index]

        # Load audio signal
        audio, lrs3_dat_samplerate = sf.read(fn_audio)

        # load video
        cap = cv2.VideoCapture(fn_video)
        frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                break
        video = np.array([frame for frame in frames]).astype(np.float32)/255
        video = np.expand_dims(video, axis=0)

        # align video and audio length
        min_len = min([ int(len(audio)/640) , video.shape[1] ])
        video  = video[:min_len]
        audio  = audio[:(min_len*640)]

        # add noise and prepare mel spectrograms
        # Load data
        if self.mode=='train':
            snr_value = torch.rand(1)*(self.SNR_range_train[1]-self.SNR_range_train[0]) + self.SNR_range_train[0]
            snr_value = snr_value.item()
        elif self.mode=='valid':
            snr_value = self.SNR_val_dedicated_valid[index]
        else:
            snr_value = self.SNR_test
        noise, noise_samplerate = sf.read(noise_fname)
        audio_n, audio, noise = self.prepare_audio_snr(audio, noise, snr_value)
        
        mel_clean = log_mel_spectrogram_whisper(audio.astype(np.float32)).T
        mel_noisy = log_mel_spectrogram_whisper(audio_n.astype(np.float32)).T

        # Get lengths for all modalities (Videolength dependent)
        min_len = min([ self.maxLenVideo, int(len(audio)/640) , video.shape[1] , int(mel_clean.shape[0]/4) ])
        video  = video[:,:min_len]
        audio  = audio[:(min_len*640)]
        mel_clean = mel_clean[:(min_len*4)]
        mel_noisy = mel_noisy[:(min_len*4)]

        # Augment video data
        if self.mode=='train':
            video = self.augment_videoframes(video)
        else:
            video = self.centreCrop_videoframes(video)

        # Augment noisy spectrum
        if self.specaugment_dict['flag']:
            mel_noisy = self.augment_spec(mel_noisy, self.specaugment_dict)
        
        
        return audio, audio_n, mel_clean, mel_noisy, video, dat_label, fid


    # Function from pystoi (https://github.com/mpariente/pystoi/blob/master/pystoi/utils.py#L131)
    def _overlap_and_add(self, x_frames: np.ndarray, hop: int) -> np.ndarray:
        """
        Perform overlap-and-add operation on a matrix of framed signal.

        Parameters:
        - x_frames (np.ndarray): Input signal frames stored in a 2D numpy array.
                                 Shape: (num_frames, frame_length)
        - hop (int): Hop size indicating the overlap between frames.

        Returns:
        - np.ndarray: Signal after overlap-and-add operation, with shape (signal_length,).
        """
        num_frames, framelen = x_frames.shape
        # Compute the number of segments, per frame.
        segments = -(-framelen // hop)  # Divide and round up.
        # Pad the framelen dimension to segments * hop and add n=segments frames
        signal = np.pad(x_frames, ((0, segments), (0, segments * hop - framelen)))
        # Reshape to a 3D tensor, splitting the framelen dimension in two
        signal = signal.reshape((num_frames + segments, segments, hop))
        # Transpose dimensions so that signal.shape = (segments, frame+segments, hop)
        signal = np.transpose(signal, [1, 0, 2])
        # Reshape so that signal.shape = (segments * (frame+segments), hop)
        signal = signal.reshape((-1, hop))
        # Now behold the magic!! Remove the last n=segments elements from the first axis
        signal = signal[:-segments]
        # Reshape to (segments, frame+segments-1, hop)
        signal = signal.reshape((segments, num_frames + segments - 1, hop))
        # This has introduced a shift by one in all rows
        # Now, reduce over the columns and flatten the array to achieve the result
        signal = np.sum(signal, axis=0)
        end = (len(x_frames) - 1) * hop + framelen
        signal = signal.reshape(-1)[:end]
        return signal


    # Function from pystoi (https://github.com/mpariente/pystoi/blob/master/pystoi/utils.py#L131)
    def remove_silent_frames(self, x, dyn_range=40, framelen=256, hop=128):
        """ 
        Function based on https://github.com/mpariente/pytorch_stoi/blob/master/torch_stoi/stoi.py
        Remove silent frames of x and y based on x
        A frame is excluded if its energy is lower than max(energy) - dyn_range
        The frame exclusion is based solely on x, the clean speech signal
        # Arguments :
            x : array, original speech wav file
            dyn_range : Energy range to determine which frame is silent (default: 40)
            framelen : Window size for energy evaluation (default: 256)
            hop : Hop size for energy evaluation (default: 128)
        # Returns :
            x without the silent frames
        """
        # Compute Mask
        w = np.hanning(framelen + 2)[1:-1]
        x_frames = np.array(
            [w * x[i:i + framelen] for i in range(0, len(x) - framelen + 1, hop)])
        # Compute energies in dB
        x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + EPS)
        # Find boolean mask of energies lower than dynamic_range dB
        # with respect to maximum clean speech energy frame
        mask = (np.max(x_energies) - dyn_range - x_energies) < 0
        # Remove silent frames by masking
        x_frames = x_frames[mask]
        x_sil = self._overlap_and_add(x_frames, hop)
        return x_sil

    
    @staticmethod
    def collate_fn(batch):
        
        audios_c, audios_n, mels_c, mels_n, videos, video_lengths, labels, fids_ = list(), list(), list(), list(), list(), list(), list(), list()
        for data in batch:
            audios_c.append(torch.tensor(data[0]))
            audios_n.append(torch.tensor(data[1]))
            mels_c.append(torch.tensor(data[2]))
            mels_n.append(torch.tensor(data[3]))
            videos.append(torch.from_numpy(data[4].copy()))
            video_lengths.append(data[4].shape[1])
            labels.append(data[5])
            fids_.append(data[6])
        
        # Padding for melspecs and videos
        maxVidLen = max(video_lengths)
        for i in range(len(video_lengths)):
            pads_mel = maxVidLen*4 - mels_c[i].shape[0]
            mels_c[i] = F.pad(mels_c[i], (0,0,0,pads_mel), "constant", 0)
            mels_n[i] = F.pad(mels_n[i], (0,0,0,pads_mel), "constant", 0)
            pads_vid = maxVidLen - videos[i].shape[1]
            videos[i] = F.pad(videos[i], (0,0,0,0,0,pads_vid), "constant", 0)
            
        return audios_c, audios_n, mels_c, mels_n, videos, labels, video_lengths, fids_

