
# Preparation


Before preparation you need to download the [LRS3-Ted](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) and the [Musan](http://www.openslr.org/17/) dataset. Then follow the next steps:

## 1. Create file.list und label.list
To start the file.list and label.list generation:
```shell
python3 create_filelist_labellist.py --lrs3 <PATH_TO_LRS3> --prep_dir <OUTPUT_PATH> --splitname <SPLIT_DEFINITION>
```
<PATH_TO_LRS3> Path to raw LRS3 data\
<OUTPUT_PATH> Path for preparation (output)\
<SPLIT_DEFINITION> can be pretrain, trainval or test for LRS3 dataset.\
<br>

## 2. Split pretrain and trainval for training and validation


## 3. Extract audio


## 4. Landmark detection 3ddfaV2


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



