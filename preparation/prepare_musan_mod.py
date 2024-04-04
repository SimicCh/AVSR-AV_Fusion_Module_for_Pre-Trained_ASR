
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import tempfile
import shutil
import os, sys, subprocess, glob, re
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

def split_musan(musan_root, prep_dir, dur=10):
    wav_fns = glob.glob(f"{musan_root}/speech/*/*wav") + glob.glob(f"{musan_root}/music/*/*wav") + glob.glob(f"{musan_root}/noise/*/*wav")
    print(f"{len(wav_fns)} raw audios")
    output_dir = prep_dir
    for wav_fn in tqdm(wav_fns):
        sample_rate, wav_data = wavfile.read(wav_fn)
        assert sample_rate == 16_000 and len(wav_data.shape) == 1
        if len(wav_data) > dur * sample_rate:
            num_split = int(np.ceil(len(wav_data) / (dur*sample_rate)))
            for i in range(num_split):
                filename = '/'.join(wav_fn.split('/')[-3:])[:-4]
                output_wav_fn = os.path.join(output_dir, filename + f'-{i}.wav')
                sub_data = wav_data[i*dur*sample_rate: (i+1)*dur*sample_rate]
                os.makedirs(os.path.dirname(output_wav_fn), exist_ok=True)
                wavfile.write(output_wav_fn, sample_rate, sub_data.astype(np.int16))
    return

def get_speaker_info(musan_root):
    wav_fns = glob.glob(f"{musan_root}/speech/*/*wav")
    spk2wav = {}
    for wav_fn in tqdm(wav_fns):
        speaker = '-'.join(os.path.basename(wav_fn).split('-')[:-1])
        if speaker not in spk2wav:
            spk2wav[speaker] = []
        spk2wav[speaker].append(wav_fn)
    speakers = sorted(list(spk2wav.keys()))
    print(f"{len(speakers)} speakers")
    np.random.shuffle(speakers)
    output_dir = f"{musan_root}/speech/"
    num_train, num_valid = int(len(speakers)*0.8), int(len(speakers)*0.1)
    train_speakers, valid_speakers, test_speakers = speakers[:num_train], speakers[num_train: num_train+num_valid], speakers[num_train+num_valid:]
    for split in ['train', 'valid', 'test']:
        speakers = eval(f"{split}_speakers")
        with open(f"{output_dir}/spk.{split}", 'w') as fo:
            fo.write('\n'.join(speakers)+'\n')
    return


def mix_audio(wav_fns):
    wav_data = [wavfile.read(wav_fn)[1] for wav_fn in wav_fns]
    wav_data_ = []
    min_len = min([len(x) for x in wav_data])
    for item in wav_data:
        wav_data_.append(item[:min_len])
    wav_data = np.stack(wav_data_).mean(axis=0).astype(np.int16)
    return wav_data

def make_musan_babble(musan_root):
    babble_dir = f"{musan_root}/babble/wav/"
    num_per_mixture = 30
    sample_rate = 16_000
    num_train, num_valid, num_test = 8000, 1000, 1000
    os.makedirs(babble_dir, exist_ok=True)
    wav_fns = glob.glob(f"{musan_root}/speech/*/*wav")
    spk2wav = {}
    for wav_fn in tqdm(wav_fns):
        speaker = '-'.join(os.path.basename(wav_fn).split('-')[:-1])
        if speaker not in spk2wav:
            spk2wav[speaker] = []
        spk2wav[speaker].append(wav_fn)
    for split in ['train', 'valid', 'test']:
        speakers = [ln.strip() for ln in open(f"{musan_root}/speech/spk.{split}").readlines()]
        num_split = eval(f"num_{split}")
        wav_fns = []
        for x in speakers:
            wav_fns.extend(spk2wav[x])
        print(f"{split} -> # speaker {len(speakers)}, # wav {len(wav_fns)}")
        for i in tqdm(range(num_split)):
            np.random.seed(i)
            perm = np.random.permutation(len(wav_fns))[:num_per_mixture]
            output_fn = f"{babble_dir}/{split}-{str(i+1).zfill(5)}.wav"
            wav_data = mix_audio([wav_fns[x] for x in perm])
            wavfile.write(output_fn, sample_rate, wav_data)
    return

def count_frames(wav_fns):
    nfs = []
    for wav_fn in tqdm(wav_fns):
        sample_rate, wav_data = wavfile.read(wav_fn)
        nfs.append(len(wav_data))
    return nfs

def make_musan_tsv(musan_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = 16_000
    min_dur, max_dur = 3*sample_rate, 11*sample_rate
    part_ratios = zip(['train', 'valid', 'test'], [0.8, 0.1, 0.1])
    all_fns = {}
    nfs = f"{musan_root}/nframes.audio"
    nfs = dict([x.strip().split('\t') for x in open(nfs).readlines()])
    for category in ['babble', 'music', 'noise']:
        wav_fns = glob.glob(f"{musan_root}/{category}/*/*wav")
        target_fns = []
        for wav_fn in tqdm(wav_fns):
            dur = int(nfs[os.path.abspath(wav_fn)])
            if dur >= min_dur and dur < max_dur:
                target_fns.append(wav_fn)
        print(f"{category}: {len(target_fns)}/{len(wav_fns)}")
        all_fns[category] = target_fns
        output_subdir = f"{output_dir}/{category}"
        os.makedirs(output_subdir, exist_ok=True)
        num_train, num_valid, num_test = int(0.8*len(target_fns)), int(0.1*len(target_fns)), int(0.1*len(target_fns))
        if category in {'music', 'noise'}:
            np.random.shuffle(target_fns)
            train_fns, valid_fns, test_fns = target_fns[:num_train], target_fns[num_train:num_train+num_valid], target_fns[num_train+num_valid:]
        elif category == 'babble':
            train_fns, valid_fns, test_fns = [], [], []
            for wav_fn in target_fns:
                split_id = os.path.basename(wav_fn)[:-4].split('-')[0]
                if split_id == 'train':
                    train_fns.append(wav_fn)
                elif split_id == 'valid':
                    valid_fns.append(wav_fn)
                elif split_id == 'test':
                    test_fns.append(wav_fn)
        for x in ['train', 'valid', 'test']:
            x_fns = eval(f"{x}_fns")
            x_fns = [os.path.abspath(x_fn) for x_fn in x_fns]
            print(os.path.abspath(output_subdir), x, len(x_fns))
            with open(f"{output_subdir}/{x}.tsv", 'w') as fo:
                fo.write('\n'.join(x_fns)+'\n')
    return

def combine_manifests(input_tsv_dirs, output_dir):
    output_subdir = f"{output_dir}/all"
    os.makedirs(output_subdir, exist_ok=True)
    num_train_per_cat = 20_000
    train_fns, valid_fns, test_fns = [], [], []
    for input_tsv_dir in input_tsv_dirs:
        train_fn, valid_fn, test_fn = [ln.strip() for ln in open(f"{input_tsv_dir}/train.tsv").readlines()], [ln.strip() for ln in open(f"{input_tsv_dir}/valid.tsv").readlines()], [ln.strip() for ln in open(f"{input_tsv_dir}/test.tsv").readlines()]
        num_repeats = int(np.ceil(num_train_per_cat/len(train_fn)))
        train_fn_ = []
        for i in range(num_repeats):
            train_fn_.extend(train_fn)
        train_fn = train_fn_[:num_train_per_cat]
        train_fns.extend(train_fn)
        valid_fns.extend(valid_fn)
        test_fns.extend(test_fn)
    for x in ['train', 'valid', 'test']:
        x_fns = eval(f"{x}_fns")
        print(os.path.abspath(output_subdir), x, len(x_fns))
        with open(f"{output_subdir}/{x}.tsv", 'w') as fo:
            fo.write('\n'.join(x_fns)+'\n')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MUSAN audio preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--musan', type=str, help='MUSAN root')
    parser.add_argument('--prep_dir', type=str, help='Prepare directory')
    args = parser.parse_args()

    print(f'musan dir: {args.musan}')
    print(f'prep dir: {args.prep_dir}')
    os.makedirs(args.prep_dir, exist_ok=True)
    print()

    print(f"Split raw audio")
    split_musan(args.musan, args.prep_dir)
    print("Finished")
    print()

    print(f"Get speaker info")
    get_speaker_info(args.prep_dir)
    print("Finished")
    print()

    print(f"Mix audio")
    make_musan_babble(args.prep_dir)
    print("Finished")
    print()

    print(f"Count number of frames")
    wav_fns = glob.glob(f"{args.prep_dir}/babble/*/*wav") + glob.glob(f"{args.prep_dir}/music/*/*wav") + glob.glob(f"{args.prep_dir}/noise/*/*wav")
    nfs = count_frames(wav_fns)
    num_frames_fn = f"{args.prep_dir}/nframes.audio"
    with open(num_frames_fn, 'w') as fo:
        for wav_fn, nf in zip(wav_fns, nfs):
            fo.write(os.path.abspath(wav_fn)+'\t'+str(nf)+'\n')
    print("Finished")
    print()
    
    print(f"Make Musan manifests")
    make_musan_tsv(args.prep_dir, os.path.join(args.prep_dir, "tsv"))
    print("Finished")
    print()


    print("Combine manifests")
    input_tsv_dirs = [f"{args.prep_dir}/tsv/{x}" for x in ['noise', 'music', 'babble']]
    combine_manifests(input_tsv_dirs, os.path.join(args.prep_dir, "tsv"))
    print("Finished")
    print()


