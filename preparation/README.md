
# Preparation


Before preparation you need to download the [LRS3-Ted](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) and the [Musan](http://www.openslr.org/17/) dataset. Then follow the next steps:

## 1. Create file.list und label.list
To start the file.list and label.list generation.
```shell
python3 create_filelist_labellist.py \
   --lrs3 <PATH_TO_LRS3> \
   --prep_dir <OUTPUT_PATH> \
   --splitname <SPLIT_DEFINITION>
```
PATH_TO_LRS3 - Path to raw LRS3 data \
OUTPUT_PATH - Path for preparation (output) \
SPLIT_DEFINITION - can be pretrain, trainval or test for LRS3 dataset \
<br>

## 2. Split pretrain and trainval for training and validation
Perform train:val split on pretrain ord trainval.
```shell
python3 create_train_val_split.py \
   --prep_dir <PREPARATION_PATH> \
   --splitname <SPLIT_DEFINITION> \
   --valid_ratio <VALID_RATIO>
```
PREPARATION_PATH - Path to preparation \
SPLIT_DEFINITION - can be pretrain, trainval or test for LRS3 dataset \
VALID_RATIO - Ratio of validation part (in percent) \
<br>


## 3. Extract audio
Etract audio from .mp4 files.
```shell
python3 extract_audio.py \
   --lrs3 <PATH_TO_LRS3> \
   --prep_dir <PREPARATION_PATH> \
   --file_list <FILE_LIST> \
   --out_dir <AUDIO_DIR> \
   --rank <RANK> \
   --nshard <NUM_SHARDS>
```
PATH_TO_LRS3 - Path to raw LRS3 data \
PREPARATION_PATH - Path to preparation \
FILE_LIST - file.list (e.g. file.list.pretrain) from step 2 \
AUDIO_DIR -  Output directory for audio files \
RANK - Selected rank \
NUM_SHARDS - Number of shards \
<br>


## 4. Landmark detection 3ddfaV2
Landmark detection with 3ddfaV2.
Download and build 3ddfaV1.
```shell
git clone https://github.com/cleardusk/3DDFA_V2.git
cd 3DDFA_V2
sh ./build.sh
```
Execute landmark detection.
```shell
python3 lm_detection_3ddfaV2.py \
   --lrs3 <PATH_TO_LRS3> \
   --prep_dir <PREPARATION_PATH> \
   --file_list <FILE_LIST> \
   --out_dir <LM_DIR> \
   --config <3DDFAV2_CONFIG> \
   --rank <RANK> \
   --nshard <NUM_SHARDS>
```
PATH_TO_LRS3 - Path to raw LRS3 data \
PREPARATION_PATH - Path to preparation \
FILE_LIST - file.list (e.g. file.list.pretrain) from step 2 \
LM_DIR -  Output directory for landmark files \
3DDFAV2_CONFIG - Path to 3ddfaV2 config - Usually ./configs/mb1_120x120.yml \
RANK - Selected rank \
NUM_SHARDS - Number of shards \
<br>



    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--prep_dir', type=str, help='prep root dir')
    parser.add_argument('--file_list', type=str, help='file list')
    parser.add_argument('--out_dir', type=str, help='output dir')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--onnx', action='store_true', default=False)
    parser.add_argument('--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')


## 5. Mouth centred video cropping




Ablauf

1. Create file.list and label.list files
create_filelist_labellist.py 

2. Split pretrain and trainval for training and validation 
create_train_val_split.py

3. Extract audio
extract_audio.py

4. Landmark detection 3ddfaV2
"""
git clone https://github.com/cleardusk/3DDFA_V2.git
cd 3DDFA_V2
sh ./build.sh
"""

lm_detection_3ddfaV2.py

5. Mouth centred video cropping
mouth_cropping_3ddfaV2.py



