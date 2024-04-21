import numpy as np
import mediapipe as mp
import os
# from utils.utils import *
import pickle
from tqdm import tqdm
# import dlib
import yaml

import sys
os.chdir('./3DDFA_V2')
sys.path.append(".")
sys.path.append("./configs")

# sys.path.append("./3DDFA_V2")
# sys.path.append("./3DDFA_V2/configs")

print(os.getcwd())

from TDDFA import TDDFA
from FaceBoxes import FaceBoxes
import imageio
from utils.functions import cv_draw_landmark, get_suffix



def detect_lms(video_fn, face_boxes, tddfa, lm_type='2d_sparse'):

    # Given a video path
    fn = video_fn.split('/')[-2:]
    reader = imageio.get_reader(video_fn)

    fps = reader.get_meta_data()['fps']

    # suffix = get_suffix(video_fn)
    # video_wfp = f'examples/{fn.replace(suffix, "")}_{lm_type}.mp4'
    # writer = imageio.get_writer(video_wfp, fps=fps)


    dense_flag = lm_type in ('3d',)
    pre_ver = None

    lm_list = list()
    for i, frame in enumerate(reader):
        frame_bgr = frame[..., ::-1]  # RGB->BGR
        ver = None

        if pre_ver is None:
            boxes = face_boxes(frame_bgr)
            if len(boxes)>0:
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                # refine
                param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
            roi_box = roi_box_lst[0]

            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                if len(boxes)>0:
                    boxes = [boxes[0]]
                    param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            else:
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        # check if results are relevant
        if ver is None:
            print(f'{fn} fr{i}: No result ...')
        else:
            if (np.max(np.array(ver).T[:,0])-np.min(np.array(ver).T[:,0])) * (np.max(np.array(ver).T[:,1])-np.min(np.array(ver).T[:,1]))<1000:
                ver = None
                print(f'{fn} fr{i}: FaceBox too small ...')


        # Add data to list
        pre_ver = ver  # for tracking
        if ver is None:
            lm_list.append(None)
        else:
            lm_list.append(np.array(ver).T)

    return lm_list




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--prep_dir', type=str, help='prep root dir')
    parser.add_argument('--file_list', type=str, help='file list')
    parser.add_argument('--out_dir', type=str, help='output dir')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--onnx', action='store_true', default=False)
    parser.add_argument('--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    args = parser.parse_args()


    print('args:')
    print(args)
    print()
    
    root_orig_data = args.lrs3
    prep_dir = args.prep_dir
    file_list = args.file_list
    out_dir = args.out_dir
    config = args.config
    rank = args.rank
    nshard = args.nshard



    # 3DFFA configs
    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()


    # Create out dir
    os.makedirs(out_dir, exist_ok=True)


    # load file list
    file_list = os.path.join(prep_dir, file_list)
    with open(file_list) as f:
        fids = [line.rstrip() for line in f]
    start_id, end_id = int((len(fids)/nshard)*rank),  int((len(fids)/nshard)*(rank+1))
    print(f'Prepare data from index {start_id} to {end_id}')
    fids = fids[start_id:end_id]

    # Start loop over fids
    for fid in tqdm(fids):
        fname = os.path.join(root_orig_data, fid+'.mp4')
        outfile = os.path.join(out_dir, fid+'.pkl')
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        lms = detect_lms(fname, face_boxes, tddfa)

        with open(outfile,"wb") as f:
            pickle.dump(lms,f)


    print()
    print("Done")

