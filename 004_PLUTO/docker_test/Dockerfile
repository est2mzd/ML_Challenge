# ベースイメージとしてPyTorchの公式イメージを使用
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 作業ディレクトリを設定
WORKDIR /work

# 必要なPythonパッケージをインストール
RUN pip install --no-cache-dir numpy

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
