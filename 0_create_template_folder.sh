#!/bin/bash

if [ $# -eq 0 ]; then
    echo "引数がありません"
    exit 1
fi

mkdir ./$1
mkdir ./$1/Codes
mkdir ./$1/Data
mkdir ./$1/Docker
mkdir ./$1/Images
mkdir ./$1/Papers
touch ./$1/README.md