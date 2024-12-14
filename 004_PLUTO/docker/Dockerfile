# 基本となるイメージを選択
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 非対話型設定
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Minicondaのインストール
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# conda環境を利用するための設定
ENV PATH=/opt/conda/bin:$PATH

# conda環境の作成とアクティベーション
RUN conda create -n pluto python=3.9 -y && conda clean -a -y
SHELL ["conda", "run", "-n", "pluto", "/bin/bash", "-c"]

# 作業ディレクトリを設定
WORKDIR /work

# nuplan-devkitのインストール
RUN mkdir -p /work/Codes/ && \
    cd /work/Codes/ && \
    git clone https://github.com/motional/nuplan-devkit.git && \
    cd nuplan-devkit && \
    pip install -e . && \
    pip install -r ./requirements.txt


# PLUTOのセットアップ
RUN cd /work/Codes/ && \
    git clone https://github.com/jchengai/pluto.git && \
    cd pluto && \
    chmod +x ./script/setup_env.sh && \
    ./script/setup_env.sh

#===========================================================#
# ホストのユーザー名とIDを設定するための環境変数
ARG USERNAME
ARG USERID

# ユーザーとsudoを設定
RUN apt-get update && \
    apt-get install -y sudo && \
    useradd -m -u $USERID -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# コンテナを起動した際にシェルに入るようにエントリーポイントを設定
USER $USERNAME
ENTRYPOINT ["/bin/bash"]
#===========================================================#

# コンテナ起動時にpluto環境をアクティブにする
# CMD ["conda", "run", "--no-capture-output", "-n", "pluto", "/bin/bash"]
