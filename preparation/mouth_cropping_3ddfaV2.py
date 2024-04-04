import numpy as np
import mediapipe as mp
import os
from utils.utils import *
import pickle
from tqdm import tqdm
import soundfile
import cv2
from utils.utils_mouth_cropping import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prep_dir', type=str, help='prep root dir')
    parser.add_argument('--landmark_dir', type=str, help='prep root dir')
    parser.add_argument('--video_dir', type=str, help='prep root dir')
    parser.add_argument('--out_dir', type=str, help='prep root dir')
    parser.add_argument('--file_list', type=str, help='file list')
    parser.add_argument('--meanface_path', type=str, help='Meanface')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    parser.add_argument('--start_idx', type=int, help='number of shards')
    parser.add_argument('--stop_idx', type=int, help='number of shards')
    parser.add_argument('--window_margin', type=int, help='number of shards')
    parser.add_argument('--crop_height', type=int, help='number of shards')
    parser.add_argument('--crop_width', type=int, help='number of shards')
    args = parser.parse_args()

    """
    root_orig_data = "/nfs/data/LRS3_Dataset"
    landmark_dir = "/net/ml1/mnt/md0/scratch/staff/simicch/00_data/02_LRS3/prep_data2"
    file_list = "file.list.trainval"
    """

    print(args)
    prep_dir = args.prep_dir
    landmark_dir = args.landmark_dir
    video_dir = args.video_dir
    out_dir = args.out_dir
    file_list = args.file_list
    meanface_path = args.meanface_path
    rank = args.rank
    nshard = args.nshard

    start_idx = args.start_idx
    stop_idx = args.stop_idx
    window_margin = args.window_margin
    crop_height = args.crop_height
    crop_width = args.crop_width

    os.makedirs(out_dir, exist_ok=True)

    file_list = os.path.join(prep_dir, file_list)
    with open(file_list) as f:
        fids = [line.rstrip() for line in f]

    start_id, end_id = int((len(fids)/nshard)*rank),  int((len(fids)/nshard)*(rank+1))
    print(f'Prepare data from index {start_id} to {end_id}')
    fids = fids[start_id:end_id]
    for fid in tqdm(fids):
        fname = os.path.join(landmark_dir, fid+'.pkl')

        video_pathname = os.path.join(video_dir, fid+'.mp4')
        landmarks_pathname = os.path.join(landmark_dir, fid+'.pkl')
        dst_pathname = os.path.join(out_dir, fid+'.mp4')

        assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)
        assert os.path.isfile(landmarks_pathname), "File does not exist. Path input: {}".format(landmarks_pathname)

        if os.path.exists(dst_pathname):
            continue


        # -- mean face utils
        STD_SIZE = (256, 256)
        mean_face_landmarks = np.load(meanface_path)
        stablePntsIDs = [33, 36, 39, 42, 45]

        landmarks = pickle.load(open(landmarks_pathname, 'rb'))
        lms_mod = list()
        for lm in landmarks:
            if lm is None:
                lms_mod.append(None)
            else:
                lms_mod.append(lm[:,:2])
        landmarks = lms_mod

        # -- pre-process landmarks: interpolate frames not being detected.
        preprocessed_landmarks = landmarks_interpolate(landmarks)

        if not preprocessed_landmarks:
            print(f"resizing {fid}")
            print(f'No landmarks for {fname}')
            frame_gen = read_video(video_pathname)
            frames = [cv2.resize(x, (crop_width, crop_height)) for x in frame_gen]
            os.makedirs(os.path.dirname(dst_pathname), exist_ok=True)
            #write_video_ffmpeg(frames, dst_pathname, args.ffmpeg)
            write_video_moviepy(frames, dst_pathname)
            continue

        # -- crop
        sequence = crop_patch(video_pathname, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                                window_margin=window_margin, 
                                start_idx=start_idx, stop_idx=stop_idx, 
                                crop_height=crop_height, crop_width=crop_width)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # -- save
        os.makedirs(os.path.dirname(dst_pathname), exist_ok=True)
        #write_video_ffmpeg(sequence, dst_pathname, args.ffmpeg)
        write_video_moviepy(sequence, dst_pathname)


print('Done')

