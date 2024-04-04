# ASR2AVSR_AV_Fusion

This is the official repo for the paper [Self-Supervised Adaptive AV Fusion Module for Pre-Trained ASR Models](https://arxiv.org/abs/2312.13873).

# About
Our approach to audio-visual speech recognition (AVSR) builds on a pre-trained ASR model (in this repo [Whisper OpenAI](https://github.com/openai/whisper)) and extends it with an upstream audio-visual fusion module to enable the ASR model to process multimodal inputs and improve the speech recogntion under noisy conditions. Details about our approach's architecture and training procedure and a comparison to the current SOTA approach to AVSR [AV-HuBERT](https://github.com/facebookresearch/av_hubert) can be taken from our [paper](https://arxiv.org/abs/2312.13873).

<p align="center">
    <img src="/imgs/Overview.jpg" alt="Bildbeschreibung" style="width: 300px;"/>
</p>

# Usage

This repo describes the usage of our approach on the [LRS3-Ted](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) dataset with synthetic noise application from the [Musan](http://www.openslr.org/17/) dataset.

1. Data pre-processing






