import glob
import os
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--prep_dir', type=str, help='prep root dir')
    parser.add_argument('--splitname', type=str, help='Splitname - pretrain/trainval/test')
    args = parser.parse_args()

    print('args:')
    print(args)
    print()

    root_orig_data = args.lrs3
    splitname = args.splitname
    prep_dir = args.prep_dir

    os.makedirs(prep_dir, exist_ok=True)

    # File list
    directory_path = os.path.join(root_orig_data, splitname)
    mp4_files = glob.glob(directory_path + '/**/*.mp4', recursive=True)
    file_list = [os.path.splitext(os.path.relpath(mp4_file, root_orig_data))[0] for mp4_file in mp4_files]
    
    print(f'Found {len(file_list)} files')

    # Create label list
    label_list = [open(os.path.join(root_orig_data, f'{el}.txt'), 'r').read() for el in file_list]
    label_list = [el.split('\n')[0].split(":")[1].strip() for el in label_list]


    # Write files
    output_fn = os.path.join(prep_dir, f'file.list.{splitname}')
    with open(output_fn, 'w') as file:
        for item in file_list:
            file.write('%s\n' % item)

    output_fn = os.path.join(prep_dir, f'label.list.{splitname}')
    with open(output_fn, 'w') as file:
        for item in label_list:
            file.write('%s\n' % item)


print('Done')

