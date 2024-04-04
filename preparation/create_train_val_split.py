import os
import argparse
from collections import Counter
import random

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prep_dir', type=str, help='prep root dir')
    parser.add_argument('--splitname', type=str, help='Input file list')
    parser.add_argument('--valid_ratio', type=float, help='Valid split ratio')
    args = parser.parse_args()
    
    print('args:')
    print(args)
    print()

    prep_dir = args.prep_dir
    splitname = args.splitname
    valid_ratio = args.valid_ratio

    # Load file list and label list
    flist_fn = os.path.join(prep_dir, f'file.list.{splitname}')
    llist_fn = os.path.join(prep_dir, f'label.list.{splitname}')

    flist = [line.rstrip() for line in open(flist_fn)]
    llist = [line.rstrip() for line in open(llist_fn)]


    # Get unique ids and count appearance
    flist_ids = [el.split('/')[1] for el in flist]
    id_counter = Counter(flist_ids)

    # shuffle unique ids
    unique_ids = list(id_counter.keys())
    random.shuffle(unique_ids)

    # Define min number of validation examples
    min_num_validexamples = valid_ratio*len(flist)
    print(f'Min Valid examples: {min_num_validexamples}')

    # Select ids for train and val
    ids_train, ids_val = list(), list()

    counter = 0
    for id in unique_ids:
        if counter<min_num_validexamples:
            ids_val.append(id)
        else:
            ids_train.append(id)
        counter += id_counter[id]


    # Create fids for train and val
    flist_train, llist_train = list(), list()
    flist_valid, llist_valid = list(), list()
    for fid, label in zip(flist,llist):
        fid_id = fid.split('/')[1]
        if fid_id in ids_val:
            flist_valid.append(fid)
            llist_valid.append(label)
        else:
            flist_train.append(fid)
            llist_train.append(label)

    print(f'Number examples train: {len(flist_train)}')
    print(f'Number examples valid: {len(flist_valid)}')

    # Ensure that train and valid examples are unseen
    unseen_flag = True
    for fid in flist_train:
        id_fid = fid.split('/')[1]
        if id_fid in ids_val:
            unseen_flag = False
    
    if unseen_flag:
        print('Unseen True')
    else:
        print('Unseen False')


    # Write files
    print('write files ...')
    with open(os.path.join(prep_dir, f"file.list.{splitname}_train"), "w") as f:
        for el in flist_train:
            f.write(f'{el}\n')
    with open(os.path.join(prep_dir, f"label.list.{splitname}_train"), "w") as f:
        for el in llist_train:
            f.write(f'{el}\n')

    with open(os.path.join(prep_dir, f"file.list.{splitname}_valid"), "w") as f:
        for el in flist_valid:
            f.write(f'{el}\n')
    with open(os.path.join(prep_dir, f"label.list.{splitname}_valid"), "w") as f:
        for el in llist_valid:
            f.write(f'{el}\n')


print('Done')

