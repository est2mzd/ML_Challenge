# nuplan-devkit/Dockerfile  をコピーして編集

# ベースとなるDockerイメージを指定（Ubuntu 20.04）
#FROM ubuntu:20.04
FROM nvidia/cuda:12.6.3-base-ubuntu20.04

# パッケージリストを更新し、必要な基本パッケージをインストール
RUN apt-get update \
    && apt-get install -y curl gnupg2 software-properties-common default-jdk

# APTキー関連の警告を無効化する環境変数を設定
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

# 必要なGPGキーを追加し、BazelやNVIDIA Docker用のリポジトリを設定
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | apt-key add - \
    && curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list \
    && add-apt-repository "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    && apt-get update \
    && apt-get install -y \
    bazel \
    file \
    zip \
    nvidia-container-toolkit \
    software-properties-common \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Minicondaをダウンロードし、サイレントモードでインストール
ENV PATH /opt/conda/bin:$PATH
RUN curl -fsSLo Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda clean -a -y

# 作業ディレクトリの設定
#ARG NUPLAN_HOME=/nuplan_devkit

# 必要なファイルをコンテナ内にコピー
#COPY requirements.txt requirements_torch.txt environment.yml /nuplan_devkit/
COPY requirements.txt requirements_torch.txt environment.yml ./

# Conda環境をrequirementsに基づいて構築
RUN PIP_EXISTS_ACTION=w conda env create -f ./environment.yml

#===========================================================#
# ホストのユーザー名とIDを設定するための環境変数
#ENV USERNAME $USERNAME
#ENV USERID $USERID

# ユーザーとsudoを設定
RUN apt-get update && \
    apt-get install -y sudo && \
    useradd -m -u $USERID -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# コンテナを起動した際にシェルに入るようにエントリーポイントを設定
USER $USERNAME
ENTRYPOINT ["/bin/bash"]
#===========================================================#

# 必要なフォルダを作成
# RUN mkdir -p $NUPLAN_HOME/nuplan

# セットアップスクリプトやコードをコンテナ内にコピー
# COPY setup.py $NUPLAN_HOME
# COPY nuplan $NUPLAN_HOME/nuplan

# Conda環境を利用するためにデフォルトシェルを設定
# SHELLを変えて、以降のコマンドは、conda環境のコマンドを実行
# SHELL ["conda", "run", "-n", "nuplan", "/bin/bash", "-c"]

# 開発モードでPythonパッケージをインストール
# RUN bash -c "pip install -e ."

#=========================================================#
# 環境変数を設定（マップやデータのルートディレクトリ）
#ENV NUPLAN_MAPS_ROOT=/data/sets/nuplan/maps \
#    NUPLAN_DATA_ROOT=/data/sets/nuplan \
#    NUPLAN_EXP_ROOT=/data/exp/nuplan

# 必要なディレクトリを作成
#RUN bash -c 'mkdir -p {$NUPLAN_MAPS_ROOT,$NUPLAN_DATA_ROOT,$NUPLAN_EXP_ROOT}'

# S3関連の環境変数を設定
#ENV NUPLAN_DATA_ROOT $NUPLAN_DATA_ROOT
#ENV NUPLAN_MAPS_ROOT $NUPLAN_MAPS_ROOT
#ENV NUPLAN_DB_FILES  /data/sets/nuplan/nuplan-v1.1/splits/mini
#ENV NUPLAN_MAP_VERSION "nuplan-maps-v1.0"
#ENV NUPLAN_S3_PROFILE "default"

#=========================================================#
# S3関連の引数を設定
#ARG NUPLAN_CHALLENGE_DATA_ROOT_S3_URL
#ARG NUPLAN_CHALLENGE_MAPS_ROOT_S3_URL
#ARG NUPLAN_SERVER_S3_ROOT_URL
#ARG S3_TOKEN_DIR
#ARG NUPLAN_DATA_STORE

# 引数として取得する環境変数
#ENV NUPLAN_DATA_ROOT_S3_URL $NUPLAN_CHALLENGE_DATA_ROOT_S3_URL
#ENV NUPLAN_MAPS_ROOT_S3_URL $NUPLAN_CHALLENGE_MAPS_ROOT_S3_URL
#ENV NUPLAN_SERVER_S3_ROOT_URL $NUPLAN_SERVER_S3_ROOT_URL
#ENV S3_TOKEN_DIR $S3_TOKEN_DIR
#ENV NUPLAN_DATA_STORE $NUPLAN_DATA_STORE
#=========================================================#

# データベースファイル用のディレクトリを作成
#RUN bash -c 'mkdir -p $NUPLAN_DB_FILES'

# コンテナ起動時に実行するスクリプトを指定
#CMD ["/nuplan_devkit/nuplan/entrypoint_simulation.sh"]
