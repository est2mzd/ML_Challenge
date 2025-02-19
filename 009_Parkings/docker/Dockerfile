FROM ubuntu:22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive

# ホストと同じユーザーでログインするための環境変数
ARG USER_NAME
ARG USER_ID

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    jupyter-notebook \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリのインストール
RUN pip3 install --no-cache-dir \
    numpy \
    casadi \
    matplotlib \
    imageio \
    jupyter
        
# ユーザー作成と権限設定
RUN useradd -m -u $USER_ID -s /bin/bash $USER_NAME && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# 作業ディレクトリ
WORKDIR /home/$USER_NAME

# ユーザー切り替え
USER $USER_NAME

# コンテナ起動時のデフォルトコマンド
CMD ["/bin/bash"]
