# インストール

nuPlan devkit をインストールするための手順を説明します。  
概要については、まず[一般的なREADME](https://github.com/motional/nuplan-devkit#readme)をご覧ください。

- [ダウンロード](#ダウンロード)
- [Pythonのインストール](#pythonのインストール)
- [仮想環境のセットアップ](#仮想環境のセットアップ)
- [devkitのインストール](#devkitのインストール)
- [必要なパッケージのインストール](#必要なパッケージのインストール)
- [環境変数の設定](#環境変数の設定)

## ダウンロード
devkitをダウンロードしてフォルダ内に移動します。

```bash
cd && git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
```

上記コマンドはファイルをホームディレクトリにダウンロードします。別のディレクトリを使用することもできますが、以降のチュートリアルではホームディレクトリを前提としています。

-----

## Pythonのインストール
このdevkitはUbuntu上でPython 3.9で動作するようにテストされています。

- **Ubuntuの場合**: 適切なPythonバージョンがシステムにインストールされていない場合、以下を実行してインストールしてください。

```bash
sudo apt install python-pip
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.9
sudo apt-get install python3.9-dev
```

- **Mac OSの場合**: [公式サイト](https://www.python.org/downloads/mac-osx/)からダウンロードしてインストールしてください。

-----

## 仮想環境のセットアップ
次に、仮想環境を設定します。Condaの使用を推奨します。

### Minicondaのインストール
[公式Minicondaページ](https://conda.io/en/latest/miniconda.html)を参照してください。

### Conda環境の作成
`environment.yml`を使用して新しいConda環境を作成します。

```bash
conda env create -f environment.yml
```

### 環境を有効化
Conda環境が有効になっている場合、シェルプロンプトは以下のように表示されます：  
`(nuplan) user@computer:~$`  
以降の作業では、Conda環境が有効であることを前提とします。環境を有効化するには以下を実行してください。

```bash
conda activate nuplan 
```

仮想環境を無効化するには以下を実行します。

```bash
conda deactivate
```

-----

## devkitのインストール

### オプションA: リモートからPIPパッケージをインストール
**注意:** このオプションはまだサポートされていません。近日中に追加予定です。

**初心者向け**の簡単な方法は、以下のコマンドでPIPパッケージをインストールすることです。

```bash
pip install nuplan-devkit
```

これにより、devkitとその依存関係がすべてインストールされます。

### オプションB: ローカルからPIPパッケージをインストール
**推奨方法**として、ローカルのdevkitをPIPパッケージとしてインストールします。

```bash
pip install -e .
```

これにより、devkitとその依存関係がすべてインストールされます。  
`-e`オプション（編集モード）は任意ですが、コードがコピーされずにそのまま使用され、簡単に開発できるようになります。

### オプションC: ソースコードを直接実行
**または**、PIPパッケージを使用したくない場合は、以下を`~/.bashrc`に追加して`nuplan-devkit`ディレクトリを`PYTHONPATH`環境変数に手動で追加します。

```bash
export PYTHONPATH="${PYTHONPATH}:$HOME/nuplan-devkit"
```

これらの変更を有効化するには、以下を実行してください。

```bash
source ~/.bashrc
```

依存関係をインストールするには以下を実行します。

```bash
pip install -r requirements_torch.txt
pip install -r requirements.txt
```

-----

## デフォルトディレクトリの変更
[一般的なREADME](https://github.com/motional/nuplan-devkit/blob/master/README.md)に記載されているように、nuPlanのデフォルトディレクトリは以下の通りです。

```bash
~/nuplan/dataset    -   データセットフォルダ。読み取り専用でも可能。
~/nuplan/exp        -   実験およびキャッシュフォルダ。読み取りおよび書き込みアクセスが必要。
```

システムでこれらを変更したい場合、`~/.bashrc`に以下を設定してください。

```bash
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
```

devkitのユニットテストを実行する場合にも、この手順が必要です。

-----

これで準備完了です！
