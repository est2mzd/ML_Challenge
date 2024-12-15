# 提出チュートリアル (Submission Tutorial)

このチュートリアルでは、有効な提出物を生成し、評価サーバーに送信する方法を説明します。

---

## 前提条件

このガイドでは、[`nuplan_framework`チュートリアル](https://github.com/motional/nuplan-devkit/blob/master/tutorials/nuplan_framework.ipynb)を実施済みであり、nuPlan環境に慣れていることを前提としています。  
最初に例として提供されるプランナー（`SimplePlanner`）でこのチュートリアルを進め、その後カスタムプランナーを使用するための修正を行うことを推奨します。

また、ローカル環境にDockerがインストールされている必要があります。公式ドキュメントは[こちら](https://docs.docker.com/get-docker/)を参照してください。ローカルテストには`docker-compose`も必要で、バージョンは`1.28.0`以上である必要があります。Linux環境では以下のコマンドで最新バージョンをインストールできます。

```bash
sudo apt remove docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.9.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

---

### プロトコル仕様

クライアント/サーバー間の通信には[gRPC](https://grpc.io/)が使用されます。通信に関連する設定ファイルやコードは[`~/nuplan_devkit/nuplan/submission`](https://github.com/motional/nuplan-devkit/blob/master/nuplan/submission)にあります。

- **注意**: 提出が正常に動作するためには、プロトコルファイル（[`protos/`](https://github.com/motional/nuplan-devkit/blob/master/nuplan/submission/protos)）および自動生成ファイル（[`challenge_pb2.py`](https://github.com/motional/nuplan-devkit/blob/master/nuplan/submission/challenge_pb2.py), [`challenge_pb2_grpc.py`](https://github.com/motional/nuplan-devkit/blob/master/nuplan/submission/challenge_pb2_grpc.py)）を変更しないでください。  

- [`submission_container.py`](https://github.com/motional/nuplan-devkit/blob/master/nuplan/submission/submission_container.py)や[`submission_planner.py`](https://github.com/motional/nuplan-devkit/blob/master/nuplan/submission/submission_planner.py)も変更しないでください。無効な提出物になる可能性があります。

---

## 有効な提出物の作成

プランナーを実行するには、[AbstractPlanner](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/planner/abstract_planner.py)を継承し、必要なインターフェースを実装する必要があります。

1. **シミュレーションのテスト**  
   カスタムプランナーを使用して[`run_simulation`スクリプト](https://github.com/motional/nuplan-devkit/blob/master/tutorials)でシミュレーションを実行します。

2. **Dockerコンテナの作成**  
   提出物としてプランナーをリモート実行するには、Dockerコンテナにパッケージ化する必要があります。

- **手順**:
  1. [`Dockerfile.submission`](https://github.com/motional/nuplan-devkit/blob/master/Dockerfile.submission)を起点として編集します。システム依存関係の編集が必要な場合は、`Dockerfile.submission`に`apt`ターゲットを追加してください。
  2. 必要な追加の`pip`パッケージがある場合は、[`requirements_submission.txt`](https://github.com/motional/nuplan-devkit/blob/master/requirements_submission.txt)に追記してください。
  3. チェックポイントや他のファイルをコンテナ内にコピーするには、`Dockerfile.submission`内の例に従ってください。
  4. [`run_submission_planner.py`](https://github.com/motional/nuplan-devkit/blob/master/nuplan/submission/run_submission_planner.py)でプランナーをインスタンス化します。

- **Dockerイメージの作成**  
   `Dockerfile.submission`を使用してDockerイメージを作成します。

```bash
docker build --network host -f Dockerfile.submission . -t nuplan-evalservice-server:test.contestant
```

3. **サーバークライアントアーキテクチャのテスト**  
   `docker-compose`を使用して動作を確認します。

```bash
docker-compose up --build
```

シミュレーション結果は、`$NUPLAN_EXP_ROOT`ディレクトリで確認できます。

---

## 提出の方法

提出するには、[EvalAI](https://eval.ai/)に登録し、CLIをインストールして以下のコマンドで提出します。

```bash
evalai push <image>:<tag> --phase random-dev-1856
```

競技の[提出ページ](https://eval.ai/web/challenges/challenge-page/1856/submission)に詳細なコマンドが表示されます。結果はEvalAIのリーダーボードに表示されます。
