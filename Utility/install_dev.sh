#!/bin/bash

sudo apt update
sudo apt install $1

# $0: スクリプトの名前
# $1: 最初の引数
# $2: 2番目の引数
# $@: すべての引数
# $#: 引数の数