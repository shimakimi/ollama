FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch torchvision torchaudio
RUN pip3 install transformers datasets accelerate bitsandbytes