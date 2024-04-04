
import os
from tqdm import tqdm
from utils.utils import extract_audio_from_video
import soundfile


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--prep_dir', type=str, help='prep root dir')
    parser.add_argument('--file_list', type=str, help='file list')
    parser.add_argument('--out_dir', type=str, help='output dir')
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
    rank = args.rank
    nshard = args.nshard

    os.makedirs(out_dir, exist_ok=True)

    file_list = os.path.join(prep_dir, file_list)
    with open(file_list) as f:
        fids = [line.rstrip() for line in f]

    start_id, end_id = int((len(fids)/nshard)*rank),  int((len(fids)/nshard)*(rank+1))
    print(f'Prepare data from index {start_id} to {end_id}')
    fids = fids[start_id:end_id]
    for fid in tqdm(fids):
        fname = os.path.join(root_orig_data, fid+'.mp4')
        outfile = os.path.join(out_dir, fid+'.wav')
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        _ = extract_audio_from_video(fname, outfile, 16000)


print('Done')
