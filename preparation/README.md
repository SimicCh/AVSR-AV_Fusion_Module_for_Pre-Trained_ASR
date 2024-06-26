
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
PREPARATION_PATH - Path to preparation directory \
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
PREPARATION_PATH - Path to preparation directory \
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
PREPARATION_PATH - Path to preparation directory \
FILE_LIST - file.list (e.g. file.list.pretrain) from step 2 \
LM_DIR -  Output directory for landmark files \
3DDFAV2_CONFIG - Path to 3ddfaV2 config - Usually ./configs/mb1_120x120.yml \
RANK - Selected rank \
NUM_SHARDS - Number of shards \
<br>

## 5. Mouth centred video cropping
Video mouth cropping with prepared landmarks.
```shell
python3 lm_detection_3ddfaV2.py \
   --lrs3 <PATH_TO_LRS3> \
   --prep_dir <PREPARATION_PATH> \
   --landmark_dir <LM_DIR> \
   --out_dir <CROPPED_DIR> \
   --file_list <FILE_LIST> \
   --rank <RANK> \
   --nshard <NUM_SHARDS>
```
PATH_TO_LRS3 - Path to raw LRS3 data \
PREPARATION_PATH - Path to preparation directory \
LM_DIR - Directory with landmark files \
CROPPED_DIR - Output directory for mouth region cropped video files \
FILE_LIST - file.list (e.g. file.list.pretrain) from step 2 \
RANK - Selected rank \
NUM_SHARDS - Number of shards \
<br>


## 6. Musan noise
Prepare noise files and file.lists for musan.
```shell
python3 prepare_musan_mod.py \
   --musan <PATH_TO_MUSAN> \
   --prep_dir <PREPARATION_PATH>
```
PATH_TO_MUSAN - Path to raw musan \
PREPARATION_PATH - Path to musan preparation directory (output) \



