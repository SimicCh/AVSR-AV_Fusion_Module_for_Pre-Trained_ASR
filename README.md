# ASR2AVSR_AV_Fusion

This is the official repo for the paper [Self-Supervised Adaptive AV Fusion Module for Pre-Trained ASR Models](https://arxiv.org/abs/2312.13873).

# About
Our approach to audio-visual speech recognition (AVSR) builds on a pre-trained ASR model (in this repo [Whisper OpenAI](https://github.com/openai/whisper)) and extends it with an upstream audio-visual fusion module to enable the ASR model to process multimodal inputs and improve the speech recogntion under noisy conditions. Details about our approach's architecture and training procedure and a comparison to the current SOTA approach to AVSR [AV-HuBERT](https://github.com/facebookresearch/av_hubert) can be taken from our [paper](https://arxiv.org/abs/2312.13873).

<p align="center">
    <img src="/imgs/Overview.jpg" alt="Bildbeschreibung" style="width: 300px;"/>
</p>

# Usage

## Preparation

Clone Repo and prepare a python virtual environment
```shell
git clone https://github.com/SimicCh/ASR2AVSR_AV_Fusion_fft.git

cd ASR2AVSR_AV_Fusion_fft

python3 -m venv .venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

This repo describes the usage of our approach on the [LRS3-Ted](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) dataset with synthetic noise application from the [Musan](http://www.openslr.org/17/) dataset. For data pre-processing follow the instructions at [preparation](./preparation/).


## Training
We follow a three stage training strategy. Prepared config files for each training stage and Whisper baseline model can be found in [configs](./configs/).
| Config file | description |
|----------|----------|
| Zeile 1  | Zeile 1  |
| Zeile 2  | Zeile 2  |
| Zeile 3  | Zeile 3  |
- **{Whisper_model}__train01_pretrain_AV_Fusion.yaml**: AV fusion module pre-training using only Lmel and Lenc on 400h pretrain split
- **{Whisper_model}__train02_finetune_AV_Fusion_base_en.yaml**: AV fusion module fine-tuning using Lmel, Lenc and Ldec on 30h trainval split
- **{Whisper_model}__train03_fullfinetune_AV_Fusion_Whisper.yaml**: AV fusion module and Whisper full-fine-tuning using Lmel, Lenc and Ldec on 30h trainval split with learning rate decay
- **{Whisper_model}__test_01_fintune_FusionOnly.yaml**: Test configuration for fine-tuned AV fusion module without Whisper fine-tuning
- **{Whisper_model}__test_01_fintune_FusionOnly_V02.yaml**: Test configuration for full-fine-tuned AV fusion and Whisper model











