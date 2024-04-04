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

<table>
  <thead>
    <tr>
      <th rowspan="2">Config file</th>
      <th rowspan="2">description</th>
      <th colspan="2">Pre-trained checkpoints</th>
    </tr>
    <tr>
      <th>AV Fusion</th>
      <th>Whisper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>base_en__train01_pretrain_AV_Fusion.yaml</td>
      <td>AV fusion module pre-training using only Lmel and Lenc on 400h pretrain split (Whisper base.en)</td>
      <td>--</td>
      <td>--</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td>base_en__train02_finetune_AV_Fusion.yaml</td>
      <td>AV fusion module fine-tuning using Lmel, Lenc and Ldec on 30h trainval split (Whisper base.en)</td>
      <td>--</td>
      <td>--</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td>base_en__train03_fullfinetune_AV_Fusion_Whisper.yaml</td>
      <td>AV fusion module and Whisper full-fine-tuning using Lmel, Lenc and Ldec on 30h trainval split with learning rate decay (Whisper base.en)</td>
      <td>--</td>
      <td>--</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td>small_en__train01_pretrain_AV_Fusion.yaml</td>
      <td>AV fusion module pre-training using only Lmel and Lenc on 400h pretrain split (Whisper small.en)</td>
      <td>--</td>
      <td>--</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td>small_en__train02_finetune_AV_Fusion.yaml</td>
      <td>AV fusion module fine-tuning using Lmel, Lenc and Ldec on 30h trainval split (Whisper small.en)</td>
      <td>--</td>
      <td>--</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td>small_en__train03_fullfinetune_AV_Fusion_Whisper.yaml</td>
      <td>AV fusion module and Whisper full-fine-tuning using Lmel, Lenc and Ldec on 30h trainval split with learning rate decay (Whisper small.en)</td>
      <td>--</td>
      <td>--</td>
    </tr>
  </tbody>
</table>

To start the training:

```shell
python train.py <path_to_config_file>
```

We recommend the following training steps:
1. {Whisper_model}__train01_pretrain_AV_Fusion.yaml - To pre-train the AV fusion module on a huge number of examples for two epochs
2. {Whisper_model}__train02_finetune_AV_Fusion.yaml - Fine-tune the AV fusion module with backpropagated information from Whisper encoder and decoder

To fine-tune only the AV fusion module:

3a. {Whisper_model}__train02_finetune_AV_Fusion.yaml - Set the learning rate decay 'lr_decay_per_epoch' to 10<sup>1/4</sup>

## Testing

For testing trained models we provided config files in [configs](./configs/).
<table>
  <thead>
    <tr>
      <th>Config file</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>base_en__test_01_fintune_FusionOnly.yaml</td>
      <td>Test training results for config base_en__train02_finetune_AV_Fusion.yaml</td>
  </tbody>
  <tbody>
    <tr>
      <td>base_en__test_02_fullfintune_inclWhisper.yaml</td>
      <td>Test training results for config base_en__train03_fullfinetune_AV_Fusion_Whisper.yaml</td>
  </tbody>
  <tbody>
    <tr>
      <td>small_en__test_01_fintune_FusionOnly.yaml</td>
      <td>Test training results for config small_en__train02_finetune_AV_Fusion.yaml</td>
  </tbody>
  <tbody>
    <tr>
      <td>small_en__test_02_fullfintune_inclWhisper.yaml</td>
      <td>Test training results for config small_en__train03_fullfinetune_AV_Fusion_Whisper.yaml</td>
  </tbody>
</table>

To start testing:

```shell
python test.py <path_to_config_file>
```



