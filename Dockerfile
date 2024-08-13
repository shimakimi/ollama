FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive

# システムの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Pythonのバージョンを確認
RUN python3 --version

# pip3の更新
RUN pip3 install --upgrade pip

# PyTorchとその他の必要なライブラリのインストール
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install transformers datasets accelerate bitsandbytes peft

# 作業ディレクトリの設定
WORKDIR /app

# コンテナ起動時に実行されるコマンド
CMD ["/bin/bash"]