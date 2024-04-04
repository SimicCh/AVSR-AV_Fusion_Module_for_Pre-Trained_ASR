import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import os
import time
from decimal import Decimal
import whisper
from whisper.decoding import DecodingTask
import jiwer
from jiwer import wer
import string

from models.AV_Fusion_model import AV_Fusion
from dataset.dataset_LRS3 import dataset_LRS3



def prepare_specaugment_dict(config):
    return  {'flag': config['dataset']['specaug_flag'],
             'sequMaxLength': config['dataset']['specaug_sequmaxlen'],
             'maxSpecLenRatio': config['dataset']['specaug_maxlenratio'],
             'maxChannels': config['dataset']['specaug_maxchannels']}


def prepare_model_dict(config, device, logfile_fn):

    model_whisper_frozen = whisper.load_model(config['model']['whisper_model'])
    for param in model_whisper_frozen.encoder.parameters():
        param.requires_grad = False
    for param in model_whisper_frozen.decoder.parameters():
        param.requires_grad = False

    model_whisper_finetune = whisper.load_model(config['model']['whisper_model'])
    for param in model_whisper_finetune.encoder.parameters():
        param.requires_grad = True
    for param in model_whisper_finetune.decoder.parameters():
        param.requires_grad = True

    model_fusion = AV_Fusion(config)

    options_whisper = whisper.DecodingOptions(language="en")

    if config['model']['whisper_chkp'] is not None:
        print(f"Load Whisper finetune model from: {config['model']['whisper_chkp']}")
        with open(logfile_fn, 'a') as log_file:
            print(f"Load Whisper finetune model from: {config['model']['whisper_chkp']}", file=log_file)
        model_whisper_finetune.load_state_dict(torch.load(config['model']['whisper_chkp']))

    if config['model']['AV_Fusion_chkp'] is not None:
        print(f"Load AV Fusion module from: {config['model']['AV_Fusion_chkp']}")
        with open(logfile_fn, 'a') as log_file:
            print(f"Load AV Fusion module from: {config['model']['AV_Fusion_chkp']}", file=log_file)
        model_fusion.load_state_dict(torch.load(config['model']['AV_Fusion_chkp']))
    
    model_whisper_frozen.to(device)
    model_whisper_finetune.to(device)
    model_fusion.to(device)

    return {
            'model_frozen': model_whisper_frozen,
            'model_finetune': model_whisper_finetune,
            'av_fusion': model_fusion,
            'options': options_whisper,
            'max_tokens': config['model']['max_tokens']
        }


def prepare_model_dict_test(config, device, logfile_fn):

    model_whisper_frozen = whisper.load_model(config['model']['whisper_model'])
    model_whisper_finetune = whisper.load_model(config['model']['whisper_model'])
    model_fusion = AV_Fusion(config)

    options_whisper = whisper.DecodingOptions(language="en")

    if config['model']['whisper_chkp'] is not None:
        print(f"Load Whisper finetune model from: {config['model']['whisper_chkp']}")
        with open(logfile_fn, 'a') as log_file:
            print(f"Load Whisper finetune model from: {config['model']['whisper_chkp']}", file=log_file)
        model_whisper_finetune.load_state_dict(torch.load(config['model']['whisper_chkp']))

    if config['model']['AV_Fusion_chkp'] is not None:
        print(f"Load AV Fusion module from: {config['model']['AV_Fusion_chkp']}")
        with open(logfile_fn, 'a') as log_file:
            print(f"Load AV Fusion module from: {config['model']['AV_Fusion_chkp']}", file=log_file)
        model_fusion.load_state_dict(torch.load(config['model']['AV_Fusion_chkp']))
    
    model_whisper_frozen.to(device)
    model_whisper_finetune.to(device)
    model_fusion.to(device)

    return {
            'model_frozen': model_whisper_frozen,
            'model_finetune': model_whisper_finetune,
            'av_fusion': model_fusion,
            'options': options_whisper,
            'max_tokens': config['model']['max_tokens']
        }




def decodingTask_main_loop(decTask: DecodingTask, audio_features: Tensor) -> Tuple[Tensor, Tensor, List[float], Tensor]:
    """
    Main loop for decoding task.
    Taken from Whisper code (modified)
    Generate tokens and logits for target model (frozen Whisper Model)
    
    Args:
        decTask (DecodingTask): Whisper decoding task containing options and model.
        audio_features (Tensor): Tensor containing audio features.
        
    Returns:
        tokens (Tensor): Tensor containing generated tokens.
        sum_logprobs (Tensor): Tensor containing sum of log probabilities.
        no_speech_probs (List[float]): List of probabilities when no speech is detected.
        logits_data (Tensor): Tensor containing logits data.
    """

    decTask.decoder.reset()
    decTask.inference.cleanup_caching()
    n_audio: int = audio_features.shape[0]
    tokens: Tensor = torch.tensor([decTask.initial_tokens]).repeat(n_audio, 1).to(audio_features.device)
    assert audio_features.shape[0] == tokens.shape[0]
    n_batch = tokens.shape[0]
    sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
    no_speech_probs = [np.nan] * n_batch

    logits_data = list()

    try:
        for i in range(decTask.sample_len):
            logits = decTask.inference.logits(tokens, audio_features)

            if (
                i == 0 and decTask.tokenizer.no_speech is not None
            ):  # save no_speech_probs
                probs_at_sot = logits[:, decTask.sot_index].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, decTask.tokenizer.no_speech].tolist()

            # now we need to consider the logits at the last token only
            logits = logits[:, -1]
            logits_data.append(logits.clone())

            # apply the logit filters, e.g. for suppressing or applying penalty to
            for logit_filter in decTask.logit_filters:
                logit_filter.apply(logits, tokens)

            # expand the tokens tensor with the selected next tokens
            tokens, completed = decTask.decoder.update(tokens, logits, sum_logprobs)

            if completed or tokens.shape[-1] > decTask.n_ctx:
                break
    finally:
        decTask.inference.cleanup_caching()
    
    logits_data = torch.stack(logits_data, dim=0)

    return tokens, sum_logprobs, no_speech_probs, logits_data


def decodingTask_main_loop_calc_Logits(decTask: DecodingTask, audio_features: Tensor, tokens: Tensor) -> Tensor:
    """
    Calculate logits data in the decoding task main loop.
    
    Args:
        decTask (DecodingTask): Whisper decoding task containing options and model.
        audio_features (Tensor): Tensor containing audio features.
        tokens (Tensor): Tensor containing generated tokens.
        
    Returns:
        logits_data (Tensor): Tensor containing logits data.
    """

    decTask.decoder.reset()
    decTask.inference.cleanup_caching()
    assert audio_features.shape[0] == tokens.shape[0]

    init_token_len = len(decTask.initial_tokens)
    logits_data = list()

    try:
        for i in range(init_token_len, tokens.shape[1]):
            # print(tokens[:,:i])
            logits = decTask.inference.logits(tokens[:,:i], audio_features)

            if (
                i == 0 and decTask.tokenizer.no_speech is not None
            ):  # save no_speech_probs
                probs_at_sot = logits[:, decTask.sot_index].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, decTask.tokenizer.no_speech].tolist()

            # now we need to consider the logits at the last token only
            logits = logits[:, -1]
            logits_data.append(logits.clone())
    finally:
        decTask.inference.cleanup_caching()

    logits_data = torch.stack(logits_data, dim=0)

    return logits_data



def log_and_save_iteration(config,
                           logfile_fn: str,
                           epoch: int,
                           iteration: int,
                           duration: float,
                           model_dict: Dict[str, torch.nn.Module],
                           loss_data: Dict[str, List]):
    
    # Logging
    txt  = f"Epoch {epoch} iteration {iteration}"
    txt += " - duration {:.2F}".format(Decimal(float(duration)))
    txt += " - Lmel {:.4F}".format(Decimal(float(np.mean(loss_data['mel'][-config['training']['log_frequency']:]))))
    txt += " - Lenc {:.4F}".format(Decimal(float(np.mean(loss_data['enc'][-config['training']['log_frequency']:]))))
    txt += " - Ldec {:.4F}".format(Decimal(float(np.mean(loss_data['dec'][-config['training']['log_frequency']:]))))

    print(txt)
    with open(logfile_fn, 'a') as log_file:
        print(txt, file=log_file)
    

    # Save checkpoints
    fname_fusion = f"{config['experiment_name']}_avfusion_epoch{epoch}_it{iteration}.pt"
    export_file_fusion = os.path.join(config['training']['work_dir'], fname_fusion)
    torch.save(model_dict['av_fusion'].state_dict(), export_file_fusion)

    if config['training']['train_whisper']:
        fname_whisper = f"{config['experiment_name']}_whisper_epoch{epoch}_it{iteration}.pt"
        export_file_whisper = os.path.join(config['training']['work_dir'], fname_whisper)
        torch.save(model_dict['model_finetune'].state_dict(), export_file_whisper)
    
    # Delete last iteration saves
    fname_fusion_old = f"{config['experiment_name']}_avfusion_epoch{epoch}_it{iteration-config['training']['log_frequency']}.pt"
    fname_fusion_old = os.path.join(config['training']['work_dir'], fname_fusion_old)
    if os.path.exists(fname_fusion_old):
        os.remove(fname_fusion_old)

    if config['training']['train_whisper']:
        fname_whisper_old = f"{config['experiment_name']}_whisper_epoch{epoch}_it{iteration-config['training']['log_frequency']}.pt"
        fname_whisper_old = os.path.join(config['training']['work_dir'], fname_whisper_old)
        if os.path.exists(fname_whisper_old):
            os.remove(fname_whisper_old)
        
    return True
    

def log_and_save_epoch(config,
                       logfile_fn: str,
                       epoch: int,
                       duration: float,
                       model_dict: Dict[str, torch.nn.Module],
                       loss_data: Dict[str, List]):
    
    # Logging
    txt  = f"Epoch {epoch} Final"
    txt += " - duration {:.2F}".format(Decimal(float(duration)))
    txt += " - Lmel {:.4F}".format(Decimal(float(np.mean(loss_data['mel']))))
    txt += " - Lenc {:.4F}".format(Decimal(float(np.mean(loss_data['enc']))))
    txt += " - Ldec {:.4F}".format(Decimal(float(np.mean(loss_data['dec']))))

    print(txt)
    with open(logfile_fn, 'a') as log_file:
        print(txt, file=log_file)
    

    # Save checkpoints
    fname_fusion = f"{config['experiment_name']}_avfusion_epoch{epoch}_Final.pt"
    export_file_fusion = os.path.join(config['training']['work_dir'], fname_fusion)
    torch.save(model_dict['av_fusion'].state_dict(), export_file_fusion)

    if config['training']['train_whisper']:
        fname_whisper = f"{config['experiment_name']}_whisper_epoch{epoch}_Final.pt"
        export_file_whisper = os.path.join(config['training']['work_dir'], fname_whisper)
        torch.save(model_dict['model_finetune'].state_dict(), export_file_whisper)
        
    return True


def log_valid_epoch(config,
                    logfile_fn: str,
                    epoch: int,
                    duration: float,
                    model_dict: Dict[str, torch.nn.Module],
                    loss_data: Dict[str, List]):
    
    # Logging
    txt  = f"Validation Epoch {epoch}"
    txt += " - duration {:.2F}".format(Decimal(float(duration)))
    txt += " - Lmel {:.4F}".format(Decimal(float(np.mean(loss_data['mel']))))
    txt += " - Lenc {:.4F}".format(Decimal(float(np.mean(loss_data['enc']))))
    txt += " - Ldec {:.4F}".format(Decimal(float(np.mean(loss_data['dec']))))

    print(txt)
    with open(logfile_fn, 'a') as log_file:
        print(txt, file=log_file)
    
    return True
    



def train_batch(batch: Tuple,
                model_dict: Dict[str, torch.nn.Module],
                optimizer: torch.optim.Optimizer,
                criterion_dict: Dict[str, torch.nn.Module],
                lossfactor_dict: Dict[str, float],
                device: torch.device,
                padding_length: int = 3000) -> Tuple[float, float, float]:
    """
    Function to train model on a batch of data.

    Args:
    - batch (Tuple): Tuple containing batch data
    - model_dict (Dict): Dictionary containing Whisper models, AV-Fusion model and Whisper options
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - criterion_dict (Dict[str, torch.nn.Module]): Dictionary containing loss functions.
    - lossfactor_dict (Dict[str, float]): Dictionary containing loss factors for different model levels.
    - padding_length (int): Length for padding mel spectra (default is 3000).

    Returns:
    - Tuple[float, float, float]: Tuple containing mel loss, audio feature encoder loss, and decoder loss.
    """

    optimizer.zero_grad()

    # get data from batch
    mels_c = torch.stack(batch[2]).to(device)
    mels_n = torch.stack(batch[3]).to(device)
    videos = torch.stack(batch[4]).to(device)
    vid_lengths = batch[6]

    # **** AV Fusion and preparation for WHisper ****
    # Melspectrum prediction by fusion model
    mels_p = model_dict['av_fusion'](mels_n, videos)[0]

    # Mespectrum padding to generate input block size of 30s
    pad_add = padding_length-mels_c.shape[1]
    if pad_add>0:
        mels_c = F.pad(input=mels_c, pad=(0,0,0,pad_add), mode='constant', value=0)
        mels_p = F.pad(input=mels_p, pad=(0,0,0,pad_add), mode='constant', value=0)
    mels_c = mels_c[:,:padding_length].permute(0, 2, 1).to(device)
    mels_p = mels_p[:,:padding_length].permute(0, 2, 1).to(device)

    # Set padding values to value 0
    for idx in range(len(vid_lengths)):
        mels_c[idx,:,vid_lengths[idx]*4:] = 0
        mels_p[idx,:,vid_lengths[idx]*4:] = 0

    # **** Whisper Encoder (Audio feature embedding level) ****
        with torch.no_grad():
            audio_features_c = model_dict['model_frozen'].encoder(mels_c)
        audio_features_p = model_dict['model_finetune'].encoder(mels_p)


    # **** Whisper Decoder (Logits level) - only if decoder loss is reqired for loss calculation ****
    if lossfactor_dict['decoder']>0:
        logits_data_c, logits_data_n, logits_data_p = list(), list(), list()
        for idx in range(audio_features_c.shape[0]):
            audio_features_c_ = audio_features_c[idx:idx+1]
            audio_features_p_ = audio_features_p[idx:idx+1]
            
            # Calc target logits from clean input
            with torch.no_grad():
                decT = DecodingTask(model_dict['model_frozen'], model_dict['options'])
                if model_dict['max_tokens'] is not None:
                    decT.n_ctx = model_dict['max_tokens']
                tokens_c, sum_logprobs, no_speech_probs, logits_data_c_ = decodingTask_main_loop(decTask=decT, audio_features=audio_features_c_)
            
            decT_ft = DecodingTask(model_dict['model_finetune'], model_dict['options'])
            if model_dict['max_tokens'] is not None:
                decT.n_ctx = model_dict['max_tokens']
            logits_data_p_ = decodingTask_main_loop_calc_Logits(decTask=decT_ft, audio_features=audio_features_p_, tokens=tokens_c)
            
            # Append to lists
            logits_data_c.append(logits_data_c_)
            logits_data_p.append(logits_data_p_)

        
        # Concatenate logits over all examples
        logits_data_c = torch.cat(logits_data_c, 0)
        logits_data_p = torch.cat(logits_data_p, 0)

        # reshape data
        logits_num = logits_data_c.shape[2]
        logits_data_c_argmax = torch.argmax(logits_data_c.view(-1, logits_num), dim=1)
        logits_data_p = logits_data_p.view(-1, logits_num)
    

    # **** Calculate Loss ****
    Loss = torch.tensor(0, dtype=torch.float64).to(device)
    if lossfactor_dict['mel']>0:
        mel_loss = [criterion_dict['l1'](mels_p[idx,:,:vid_lengths[idx]*4], mels_c[idx,:,:vid_lengths[idx]*4]) for idx in range(len(mels_c))]
        mel_loss = sum(mel_loss)/len(mel_loss)
        Loss += lossfactor_dict['mel']*mel_loss
    else:
        mel_loss = torch.tensor(0.0)
    
    if lossfactor_dict['audiofeatures']>0:
        enc_loss = criterion_dict['l1'](audio_features_p, audio_features_c)
        Loss += lossfactor_dict['audiofeatures']*enc_loss
    else:
        enc_loss = torch.tensor(0.0)
    
    if lossfactor_dict['decoder']>0:
        dec_loss = criterion_dict['crossentropy'](logits_data_p, logits_data_c_argmax)
        Loss += lossfactor_dict['decoder']*dec_loss
    else:
        dec_loss = torch.tensor(0.0)
    
    # Update step
    Loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return mel_loss.to('cpu').detach().numpy(), enc_loss.to('cpu').detach().numpy(), dec_loss.to('cpu').detach().numpy()


def train_epoch(config,
                epoch: int,
                logfile_fn: str,
                model_dict: Dict[str, torch.nn.Module],
                optimizer: torch.optim.Optimizer,
                criterion_dict: Dict[str, torch.nn.Module],
                lossfactor_dict: Dict[str, float],
                device: torch.device,
                padding_length: int = 3000):
    
    model_dict['model_frozen'].eval()
    model_dict['model_finetune'].train()
    model_dict['av_fusion'].train()
    
    specaugment_dict = prepare_specaugment_dict(config)
    
    ds_train = dataset_LRS3(lrs3_prep_dir=config['dataset']['lrs3_prep_dir'],
            lrs3_lists_path=config['dataset']['lrs3_lists_path'],
            lrs3_filelist=config['dataset']['lrs3_filelist_training'],
            lrs3_audio_dir=config['dataset']['lrs3_audio_dir'],
            lrs3_video_dir=config['dataset']['lrs3_video_dir'],
            musan_prep_dir=config['dataset']['musan_prep_dir'],
            musan_type=config['dataset']['musan_type'],
            mode=config['mode'],
            specaugment_dict=specaugment_dict,
            SNR_range_train=tuple(config['dataset']['SNR_range_training']),
            logfile_fn=logfile_fn,
            set_seed=True,
            seed=epoch,
            maxLenVideo=config['dataset']['maxLenVideo'],
            frameCropShape=tuple(config['dataset']['frameCropShape']),
            flipProb=config['dataset']['flipProb'],
            erasingProb=config['dataset']['erasingProb'],
            eraseSizeRange=tuple(config['dataset']['eraseSizeRange']),
            eraseRatioRange=tuple(config['dataset']['eraseRatioRange']),
            timeMaskRange=tuple(config['dataset']['timeMaskRange']))

    dl_train = DataLoader(
                ds_train,
                shuffle=True,
                batch_size=config['dataset']['batchsize_training'],
                num_workers=config['dataset']['batchsize_training'],
                pin_memory=True,
                drop_last=True,
                collate_fn=ds_train.collate_fn,
            )
    
    loss_data = {'mel': list(),
                 'enc': list(),
                 'dec': list()}

    t0 = time.time()
    for i, batch in enumerate(dl_train):

        mel_loss, enc_loss, dec_loss = train_batch(batch=batch,
                                                   model_dict=model_dict,
                                                   optimizer=optimizer,
                                                   criterion_dict=criterion_dict,
                                                   lossfactor_dict=lossfactor_dict,
                                                   device=device)
        
        loss_data['mel'].append(mel_loss)
        loss_data['enc'].append(enc_loss)
        loss_data['dec'].append(dec_loss)

        if (i+1)%config['training']['log_frequency'] == 0:
            log_and_save_iteration(config=config,
                                   logfile_fn=logfile_fn,
                                   epoch=epoch,
                                   iteration=i+1,
                                   duration=time.time()-t0,
                                   model_dict=model_dict,
                                   loss_data=loss_data)
    
    # Log and save final models after epoch
    log_and_save_epoch(config=config,
                       logfile_fn=logfile_fn,
                       epoch=epoch,
                       duration=time.time()-t0,
                       model_dict=model_dict,
                       loss_data=loss_data)
    
    return loss_data
    



def valid_batch(batch: Tuple,
                model_dict: Dict[str, torch.nn.Module],
                criterion_dict: Dict[str, torch.nn.Module],
                device: torch.device,
                padding_length: int = 3000) -> Tuple[float, float, float]:
    """
    Function to validate model on a batch of data.

    Args:
    - batch (Tuple): Tuple containing batch data
    - model_dict (Dict): Dictionary containing Whisper models, AV-Fusion model and Whisper options
    - criterion_dict (Dict[str, torch.nn.Module]): Dictionary containing loss functions.
    - padding_length (int): Length for padding mel spectra (default is 3000).

    Returns:
    - Tuple[float, float, float]: Tuple containing mel loss, audio feature encoder loss, and decoder loss.
    """

    # get data from batch
    mels_c = torch.stack(batch[2]).to(device)
    mels_n = torch.stack(batch[3]).to(device)
    videos = torch.stack(batch[4]).to(device)
    vid_lengths = batch[6]

    # **** AV Fusion and preparation for WHisper ****
    # Melspectrum prediction by fusion model
    mels_p = model_dict['av_fusion'](mels_n, videos)[0]

    # Mespectrum padding to generate input block size of 30s
    pad_add = padding_length-mels_c.shape[1]
    if pad_add>0:
        mels_c = F.pad(input=mels_c, pad=(0,0,0,pad_add), mode='constant', value=0)
        mels_p = F.pad(input=mels_p, pad=(0,0,0,pad_add), mode='constant', value=0)
    mels_c = mels_c[:,:padding_length].permute(0, 2, 1).to(device)
    mels_p = mels_p[:,:padding_length].permute(0, 2, 1).to(device)

    # Set padding values to value 0
    for idx in range(len(vid_lengths)):
        mels_c[idx,:,vid_lengths[idx]*4:] = 0
        mels_p[idx,:,vid_lengths[idx]*4:] = 0

    # **** Whisper Encoder (Audio feature embedding level) ****
        with torch.no_grad():
            audio_features_c = model_dict['model_frozen'].encoder(mels_c)
        audio_features_p = model_dict['model_finetune'].encoder(mels_p)


    # **** Whisper Decoder (Logits level) ****
    logits_data_c, logits_data_n, logits_data_p = list(), list(), list()
    for idx in range(audio_features_c.shape[0]):
        audio_features_c_ = audio_features_c[idx:idx+1]
        audio_features_p_ = audio_features_p[idx:idx+1]
        
        # Calc target logits from clean input
        with torch.no_grad():
            decT = DecodingTask(model_dict['model_frozen'], model_dict['options'])
            if model_dict['max_tokens'] is not None:
                decT.n_ctx = model_dict['max_tokens']
            tokens_c, sum_logprobs, no_speech_probs, logits_data_c_ = decodingTask_main_loop(decTask=decT, audio_features=audio_features_c_)
        
        decT_ft = DecodingTask(model_dict['model_finetune'], model_dict['options'])
        if model_dict['max_tokens'] is not None:
            decT.n_ctx = model_dict['max_tokens']
        logits_data_p_ = decodingTask_main_loop_calc_Logits(decTask=decT_ft, audio_features=audio_features_p_, tokens=tokens_c)
        
        # Append to lists
        logits_data_c.append(logits_data_c_)
        logits_data_p.append(logits_data_p_)

    
    # Concatenate logits over all examples
    logits_data_c = torch.cat(logits_data_c, 0)
    logits_data_p = torch.cat(logits_data_p, 0)

    # reshape data
    logits_num = logits_data_c.shape[2]
    logits_data_c_argmax = torch.argmax(logits_data_c.view(-1, logits_num), dim=1)
    logits_data_p = logits_data_p.view(-1, logits_num)
    

    # **** Calculate Loss ****
    mel_loss = [criterion_dict['l1'](mels_p[idx,:,:vid_lengths[idx]*4], mels_c[idx,:,:vid_lengths[idx]*4]) for idx in range(len(mels_c))]
    mel_loss = sum(mel_loss)/len(mel_loss)
    enc_loss = criterion_dict['l1'](audio_features_p, audio_features_c)
    dec_loss = criterion_dict['crossentropy'](logits_data_p, logits_data_c_argmax)

    return mel_loss.to('cpu').detach().numpy(), enc_loss.to('cpu').detach().numpy(), dec_loss.to('cpu').detach().numpy()


def valid_epoch(config,
                epoch: int,
                logfile_fn: str,
                model_dict: Dict[str, torch.nn.Module],
                criterion_dict: Dict[str, torch.nn.Module],
                device: torch.device,
                padding_length: int = 3000):
    
    model_dict['model_frozen'].eval()
    model_dict['model_finetune'].eval()
    model_dict['av_fusion'].eval()

    specaugment_dict = prepare_specaugment_dict(config)
    specaugment_dict['flag'] = False
    
    ds_valid = dataset_LRS3(lrs3_prep_dir=config['dataset']['lrs3_prep_dir'],
            lrs3_lists_path=config['dataset']['lrs3_lists_path'],
            lrs3_filelist=config['dataset']['lrs3_filelist_validation'],
            lrs3_audio_dir=config['dataset']['lrs3_audio_dir'],
            lrs3_video_dir=config['dataset']['lrs3_video_dir'],
            musan_prep_dir=config['dataset']['musan_prep_dir'],
            musan_type=config['dataset']['musan_type'],
            mode='valid',
            specaugment_dict=specaugment_dict,
            SNR_range_valid=tuple(config['dataset']['SNR_range_valid']),
            logfile_fn=None,
            set_seed=True,
            seed=0,
            maxLenVideo=config['dataset']['maxLenVideo'],
            frameCropShape=tuple(config['dataset']['frameCropShape']))
    ds_valid.fids = ds_valid.fids[:config['dataset']['validation_num_examples']]

    dl_valid = DataLoader(
                ds_valid,
                shuffle=True,
                batch_size=config['dataset']['batchsize_valid'],
                num_workers=config['dataset']['batchsize_valid'],
                pin_memory=True,
                drop_last=True,
                collate_fn=ds_valid.collate_fn,
            )
    
    loss_data = {'mel': list(),
                 'enc': list(),
                 'dec': list()}

    t0 = time.time()
    for i, batch in enumerate(dl_valid):

        mel_loss, enc_loss, dec_loss = valid_batch(batch=batch,
                                                   model_dict=model_dict,
                                                   criterion_dict=criterion_dict,
                                                   device=device)
        
        loss_data['mel'].append(mel_loss)
        loss_data['enc'].append(enc_loss)
        loss_data['dec'].append(dec_loss)
    
    
    # Log and save final models after epoch
    log_valid_epoch(config=config,
                    logfile_fn=logfile_fn,
                    epoch=epoch,
                    duration=time.time()-t0,
                    model_dict=model_dict,
                    loss_data=loss_data)
    
    return loss_data




def test_batch(batch: Tuple,
               model_dict: Dict[str, torch.nn.Module],
               criterion_dict: Dict[str, torch.nn.Module],
               device: torch.device,
               padding_length: int = 3000) -> Tuple[float, float, float]:
    """
    Function to test model on a batch of data.

    Args:
    - batch (Tuple): Tuple containing batch data
    - model_dict (Dict): Dictionary containing Whisper models, AV-Fusion model and Whisper options
    - criterion_dict (Dict[str, torch.nn.Module]): Dictionary containing loss functions.
    - padding_length (int): Length for padding mel spectra (default is 3000).

    Returns:
    - Tuple[float, float, float]: Tuple containing mel loss, audio feature encoder loss, and decoder loss.
    """

    # get data from batch
    mels_c = torch.stack(batch[2]).to(device)
    mels_n = torch.stack(batch[3]).to(device)
    videos = torch.stack(batch[4]).to(device)
    vid_lengths = batch[6]
    labels = batch[5]


    # **** AV Fusion and preparation for WHisper ****
    # Melspectrum prediction by fusion model
    with torch.no_grad():
        mels_p = model_dict['av_fusion'](mels_n, videos)[0]

    # Mespectrum padding to generate input block size of 30s
    pad_add = padding_length-mels_c.shape[1]
    if pad_add>0:
        mels_c = F.pad(input=mels_c, pad=(0,0,0,pad_add), mode='constant', value=0)
        mels_n = F.pad(input=mels_n, pad=(0,0,0,pad_add), mode='constant', value=0)
        mels_p = F.pad(input=mels_p, pad=(0,0,0,pad_add), mode='constant', value=0)
    mels_c = mels_c[:,:padding_length].permute(0, 2, 1).to(device)
    mels_n = mels_n[:,:padding_length].permute(0, 2, 1).to(device)
    mels_p = mels_p[:,:padding_length].permute(0, 2, 1).to(device)

    # Set padding values to value 0
    for idx in range(len(vid_lengths)):
        min_val_c = torch.min(mels_c[idx,:,:vid_lengths[idx]*4]).detach()
        min_val_n = torch.min(mels_n[idx,:,:vid_lengths[idx]*4]).detach()
        mels_c[idx,:,vid_lengths[idx]*4:] = min_val_c
        mels_n[idx,:,vid_lengths[idx]*4:] = min_val_n
        mels_p[idx,:,vid_lengths[idx]*4:] = min_val_n

    # # **** Whisper Encoder (Audio feature embedding level) ****
    #     with torch.no_grad():
    #         audio_features_c = model_dict['model_frozen'].encoder(mels_c)
    #     audio_features_p = model_dict['model_finetune'].encoder(mels_p)


    # # **** Whisper Decoder (Logits level) ****
    # logits_data_c, logits_data_n, logits_data_p = list(), list(), list()
    # for idx in range(audio_features_c.shape[0]):
    #     audio_features_c_ = audio_features_c[idx:idx+1]
    #     audio_features_p_ = audio_features_p[idx:idx+1]
    #     
    #     # Calc target logits from clean input
    #     with torch.no_grad():
    #         decT = DecodingTask(model_dict['model_frozen'], model_dict['options'])
    #         if model_dict['max_tokens'] is not None:
    #             decT.n_ctx = model_dict['max_tokens']
    #         tokens_c, sum_logprobs, no_speech_probs, logits_data_c_ = decodingTask_main_loop(decTask=decT, audio_features=audio_features_c_)
    #     
    #     decT_ft = DecodingTask(model_dict['model_finetune'], model_dict['options'])
    #     if model_dict['max_tokens'] is not None:
    #         decT.n_ctx = model_dict['max_tokens']
    #     logits_data_p_ = decodingTask_main_loop_calc_Logits(decTask=decT_ft, audio_features=audio_features_p_, tokens=tokens_c)
    #     
    #     # Append to lists
    #     logits_data_c.append(logits_data_c_)
    #     logits_data_p.append(logits_data_p_)

    # # Concatenate logits over all examples
    # logits_data_c = torch.cat(logits_data_c, 0)
    # logits_data_p = torch.cat(logits_data_p, 0)

    # # reshape data
    # logits_num = logits_data_c.shape[2]
    # logits_data_c_argmax = torch.argmax(logits_data_c.view(-1, logits_num), dim=1)
    # logits_data_p = logits_data_p.view(-1, logits_num)
    

    # **** Whisper transcriptions ****
    with torch.no_grad():
        result_c  = whisper.decode(model_dict['model_frozen'], mels_c.to(model_dict['model_frozen'].device), model_dict['options'])
        result_n  = whisper.decode(model_dict['model_frozen'], mels_n.to(model_dict['model_frozen'].device), model_dict['options'])
        result_p  = whisper.decode(model_dict['model_finetune'], mels_p.to(model_dict['model_finetune'].device), model_dict['options'])

    gt_tr = [label.lower().translate(str.maketrans('', '', string.punctuation)) for label in labels]
    tr_c  = [res.text.lower().translate(str.maketrans('', '', string.punctuation)) for res in result_c]
    tr_n  = [res.text.lower().translate(str.maketrans('', '', string.punctuation)) for res in result_n]
    tr_p  = [res.text.lower().translate(str.maketrans('', '', string.punctuation)) for res in result_p]


    # **** Calculate Loss ****
    mel_loss = [criterion_dict['l1'](mels_p[idx,:,:vid_lengths[idx]*4], mels_c[idx,:,:vid_lengths[idx]*4]) for idx in range(len(mels_c))]
    mel_loss = sum(mel_loss)/len(mel_loss)
    # enc_loss = criterion_dict['l1'](audio_features_p, audio_features_c)
    enc_loss = torch.tensor(0)
    # dec_loss = criterion_dict['crossentropy'](logits_data_p, logits_data_c_argmax)
    dec_loss = torch.tensor(0)

    transcription_dict = {'gt': gt_tr,
                          'c': tr_c,
                          'n': tr_n,
                          'p': tr_p}

    return mel_loss.to('cpu').detach().numpy(), enc_loss.to('cpu').detach().numpy(), dec_loss.to('cpu').detach().numpy(), transcription_dict


def test_epoch(config,
               logfile_fn: str,
               model_dict: Dict[str, torch.nn.Module],
               criterion_dict: Dict[str, torch.nn.Module],
               noise_category: str,
               SNR_value: int,
               device: torch.device,
               padding_length: int = 3000):
    
    # model_dict['model_frozen'].train()
    # model_dict['model_finetune'].train()
    model_dict['av_fusion'].eval()

    specaugment_dict = {'flag': False}
    
    ds_test = dataset_LRS3(
            lrs3_prep_dir=config['dataset']['lrs3_prep_dir'],
            lrs3_lists_path=config['dataset']['lrs3_lists_path'],
            lrs3_filelist=config['dataset']['lrs3_filelist'],
            lrs3_audio_dir=config['dataset']['lrs3_audio_dir'],
            lrs3_video_dir=config['dataset']['lrs3_video_dir'],
            musan_prep_dir=config['dataset']['musan_prep_dir'],
            musan_type=noise_category,
            mode='test',
            specaugment_dict=specaugment_dict,
            SNR_test=SNR_value,
            logfile_fn=None,
            set_seed=True,
            seed=1,
            frameCropShape=tuple(config['dataset']['frameCropShape']))
    # ds_test.fids = ds_test.fids[:100]

    dl_test = DataLoader(
                ds_test,
                shuffle=False,
                batch_size=config['test']['batchsize'],
                num_workers=config['test']['batchsize'],
                pin_memory=True,
                drop_last=True,
                collate_fn=ds_test.collate_fn,
            )
    
    loss_data_lists = {
            'mel': list(),
            'enc': list(),
            'dec': list()
        }
    
    trancipts = {
            'gt': list(),
            'c': list(),
            'n': list(),
            'p': list()
        }

    t0 = time.time()
    for i, batch in enumerate(dl_test):

        mel_loss, enc_loss, dec_loss, transcr_ = test_batch(batch=batch,
                                                   model_dict=model_dict,
                                                   criterion_dict=criterion_dict,
                                                   device=device)
        
        loss_data_lists['mel'].append(mel_loss)
        loss_data_lists['enc'].append(enc_loss)
        loss_data_lists['dec'].append(dec_loss)
        trancipts['gt'] += transcr_['gt']
        trancipts['c'] += transcr_['c']
        trancipts['n'] += transcr_['n']
        trancipts['p'] += transcr_['p']

    wer_meas_c = jiwer.compute_measures(trancipts['gt'], trancipts['c'])
    wer_meas_n = jiwer.compute_measures(trancipts['gt'], trancipts['n'])
    wer_meas_p = jiwer.compute_measures(trancipts['gt'], trancipts['p'])

    results = {
        'Lmel': np.mean(loss_data_lists['mel']),
        'Lenc': np.mean(loss_data_lists['enc']),
        'Ldec': np.mean(loss_data_lists['dec']),
        'WER_c': wer_meas_c['wer'],
        'WER_n': wer_meas_n['wer'],
        'WER_p': wer_meas_p['wer']
    }
    
    return results
        

def train(config):

    # Define working directory and log file name
    work_dir = config['training']['work_dir']
    os.makedirs(work_dir, exist_ok=True)
    print(f'Output directory: {work_dir}')

    logfile_fn = os.path.join(work_dir, f"{config['experiment_name']}_log_startep{config['training']['start_epoch']}.log")
    log_file = open(logfile_fn, "w")
    log_file.close()
    print(f"logfile {logfile_fn}")
    print("")


    # Create/Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("")
    with open(logfile_fn, 'a') as log_file:
        print(f"Device: {device}", file=log_file)
        print("", file=log_file)

    model_dict = prepare_model_dict(config, device, logfile_fn)


    # Define Optimizer, crits and losses
    lr = config['training']['lr']
    training_parameters = list()
    if config['training']['train_av_fusion']:
        training_parameters += list(model_dict['av_fusion'].parameters())
    if config['training']['train_whisper']:
        training_parameters += list(model_dict['model_finetune'].parameters())
    optimizer = optim.Adam(training_parameters, lr=lr)

    criterion_dict = {'l1': nn.L1Loss().to(device),
                    'crossentropy': nn.CrossEntropyLoss().to(device)}
    
    lossfactor_dict = {
                'mel': config['training']['lossfactor_mel'],
                'audiofeatures': config['training']['lossfactor_audiofeatures'],
                'decoder': config['training']['lossfactor_decoder']
            }


    # Start training
    for epoch in range(config['training']['start_epoch'],config['training']['epochs']):

        print("")
        print("******************")
        print(f"Start epoch {epoch}")
        with open(logfile_fn, 'a') as log_file:
            print("", file=log_file)
            print("******************", file=log_file)
            print(f"Start epoch {epoch}", file=log_file)

        # Reset learning rate if learning rate decay is activated
        if config['training']['lr_decay_per_epoch']>1.0:
            lr = lr/config['training']['lr_decay_per_epoch']
            for g in optimizer.param_groups:
                g['lr'] = lr
            lr = optimizer.param_groups[0]['lr']
            print(f"New learning rate: {lr}")
            with open(logfile_fn, 'a') as log_file:
                print(f"New learning rate: {lr}", file=log_file)


        ret = train_epoch(config=config,
                          epoch=epoch,
                          logfile_fn=logfile_fn,
                          model_dict=model_dict,
                          optimizer=optimizer,
                          criterion_dict=criterion_dict,
                          lossfactor_dict=lossfactor_dict,
                          device=device)

        ret = valid_epoch(config=config,
                          epoch=epoch,
                          logfile_fn=logfile_fn,
                          model_dict=model_dict,
                          criterion_dict=criterion_dict,
                          device=device)

    print('Finished Training')


def test(config):

    work_dir = config['test']['work_dir']
    os.makedirs(work_dir, exist_ok=True)
    print(f'Work directory: {work_dir}')

    logfile_fn = os.path.join(work_dir, f"{config['experiment_name']}__log.log")
    log_file = open(logfile_fn, "w")
    log_file.close()
    print(f"logfile {logfile_fn}")
    print("")


    # Create/Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("")
    with open(logfile_fn, 'a') as log_file:
        print(f"Device: {device}", file=log_file)
        print("", file=log_file)

    model_dict = prepare_model_dict_test(config, device, logfile_fn)

    criterion_dict = {'l1': nn.L1Loss().to(device),
                    'crossentropy': nn.CrossEntropyLoss().to(device)}
    
    # Specaugment not needed
    specaugment_dict = {'flag': False}

    cat_list = config['test']['noise_categories']
    snr_list = config['test']['SNR_values']

    result_matrix_n = np.zeros([len(cat_list),  len(snr_list)])
    result_matrix_p = np.zeros([len(cat_list),  len(snr_list)])

    for snr_idx, snr in enumerate(snr_list):
        print(f'\nSNR {snr}')
        with open(logfile_fn, 'a') as log_file:
            print(f'\nSNR {snr}', file=log_file)

        for cat_idx, cat in enumerate(cat_list):
            t0 = time.time()
            results = test_epoch(config=config,
                            logfile_fn=logfile_fn,
                            model_dict=model_dict,
                            criterion_dict=criterion_dict,
                            noise_category=cat,
                            SNR_value=snr,
                            device=device)

            result_matrix_n[cat_idx, snr_idx] = results['WER_n']
            result_matrix_p[cat_idx, snr_idx] = results['WER_p']

            # Log and save final models after epoch
            txt = f'  Cat {cat}'
            txt += " Duration {:.2F}".format(Decimal(float(time.time() - t0)))
            txt += " - Lmel {:.4F}".format(Decimal(float(results['Lmel'])))
            txt += " - Lenc {:.4F}".format(Decimal(float(results['Lenc'])))
            txt += " - Ldec {:.4F}".format(Decimal(float(results['Ldec'])))
            txt += " - WER_c {:.2F}%".format(Decimal(float(100*results['WER_c'])))
            txt += " - WER_n {:.2F}%".format(Decimal(float(100*results['WER_n'])))
            txt += " - WER_p {:.2F}%".format(Decimal(float(100*results['WER_p'])))

            print(txt)
            with open(logfile_fn, 'a') as log_file:
                print(txt, file=log_file)



    # Logging Final results
    print("")
    with open(logfile_fn, 'a') as log_file:
        print("", file=log_file)

    txt = 'WER [%] results n:\nSNR_values ' + " ".join(str(snr) for snr in snr_list)
    print(txt)
    with open(logfile_fn, 'a') as log_file:
        print(txt, file=log_file)

    for i in range(len(cat_list)):
        txt = f'{cat_list[i]} ' + " ".join(["{:.2F}".format(Decimal(float(100*el))) for el in result_matrix_n[i]])
        print(txt)
        with open(logfile_fn, 'a') as log_file:
            print(txt, file=log_file)

    print("")
    with open(logfile_fn, 'a') as log_file:
        print("", file=log_file)



    print("")
    with open(logfile_fn, 'a') as log_file:
        print("", file=log_file)

    txt = 'WER [%] results p:\nSNR_values ' + " ".join(str(snr) for snr in snr_list)
    print(txt)
    with open(logfile_fn, 'a') as log_file:
        print(txt, file=log_file)

    for i in range(len(cat_list)):
        txt = f'{cat_list[i]} ' + " ".join(["{:.2F}".format(Decimal(float(100*el))) for el in result_matrix_p[i]])
        print(txt)
        with open(logfile_fn, 'a') as log_file:
            print(txt, file=log_file)

    print("")
    with open(logfile_fn, 'a') as log_file:
        print("", file=log_file)


    print("Finished test.")



