<div align="center">

# nuPlan
**世界初の自動運転車プランニング用ベンチマーク**

______________________________________________________________________

<p align="center">
  <a href="https://www.nuplan.org/">公式ウェブサイト</a> •
  <a href="https://www.nuscenes.org/nuplan#download">ダウンロード</a> •
  <a href="#citation">引用情報</a><br>
  <a href="#changelog">変更履歴</a> •
  <a href="#devkit-structure">構成</a> •
  <a href="https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md">セットアップ</a> <br>
  <a href="https://github.com/motional/nuplan-devkit/blob/master/tutorials/nuplan_framework.ipynb">チュートリアル</a> •
  <a href="https://nuplan-devkit.readthedocs.io/en/latest/">ドキュメント</a> •
  <a href="https://eval.ai/web/challenges/challenge-page/1856/overview">競技</a>
</p>

[![python](https://img.shields.io/badge/python-%20%203.9-blue.svg)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/motional/nuplan-devkit/blob/master/LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/nuplan-devkit/badge/?version=latest)](https://nuplan-devkit.readthedocs.io/en/latest/?badge=latest)

______________________________________________________________________

<br>

<p align="center"><img src="https://www.nuplan.org/static/media/nuPlan_final.3fde7586.png" width="500px"></p>

</div>

______________________________________________________________________

## センサーデータリリース
#### **重要**: ファイル構造が変更されました！最新のファイル構造については[Dataset Setup](https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md)ページをご確認ください。

- v1.1データセット用のnuPlanセンサーデータがリリースされました。最新データセットは[nuPlanページ](https://www.nuscenes.org/nuplan#download)からダウンロードしてください。
- センサーデータのサイズが大きいため、段階的にリリースされます。最初のリリースでは、nuPlan miniに対応するデータが含まれています。
- センサーデータの簡単なチュートリアルとして`nuplan_sensor_data_tutorial.ipynb`が用意されています。

______________________________________________________________________

## プランニングチャレンジ
#### **重要**: nuPlan提出用のベースDockerイメージが更新されました。新しい`Dockerfile.submission`を使用して提出コンテナを再構築してください。

- プランニングチャレンジでは、devkitバージョン1.2が使用されます。v1.1で生成された提出物は引き続き互換性があるはずですが、ウォームアップフェーズで再確認することをお勧めします。
- チャレンジはCVPR 2023の[End-to-End Autonomous Driving](https://opendrivelab.com/event/cvpr23_ADworkshop)ワークショップの一環として発表されます。
- nuPlan Dataset v1.1がリリースされました。最新データセットは[こちら](https://www.nuscenes.org/nuplan#download)からダウンロードしてください。

______________________________________________________________________

## 変更履歴
- **2023年5月11日**  
  * v1.2.2 Devkit: 提出用ベースイメージを更新。
- **2023年5月9日**  
  * v1.2.1 Devkit: 競技の締切を2023年5月26日まで延長。
- **2023年4月25日**  
  * v1.2 Devkit: センサーデータがリリースされ、キャッシュ機能とnuBoardダッシュボードの機能が改善。
- **2023年1月20日**  
  * v1.1 Devkit: 公式nuPlanチャレンジのリリース。トレーニングキャッシュの最適化、シミュレーション改善。
- **2022年10月13日**  
  * v1.1 Dataset: 完全版nuPlanデータセットがリリース。

______________________________________________________________________

## Devkitとデータセットのセットアップ
詳細なセットアップ手順については[インストールページ](https://nuplan-devkit.readthedocs.io/en/latest/installation.html)をご覧ください。

データセットのダウンロードとセットアップ手順については[データセットページ](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)をご覧ください。

______________________________________________________________________

## はじめに
nuPlan Planning Competitionへの参加を希望される方は、[競技ページ](https://nuplan-devkit.readthedocs.io/en/latest/)をご参照ください。

- nuPlanの主要な[機能](https://www.nuplan.org)および[データセットの説明](https://www.nuplan.org/nuplan)に慣れてください。
- [セットアップ手順](#devkitとデータセットのセットアップ)に従い、devkitとデータセットを準備してください。
- [このフォルダ](https://github.com/motional/nuplan-devkit/blob/master/tutorials/)内のチュートリアルを実行してください。

______________________________________________________________________

## パフォーマンスチューニングガイド
トレーニング構成は、システムパフォーマンス（前処理コスト、トレーニング速度、数値的安定性など）を確保するために重要です。これらの問題に遭遇した場合は、[パフォーマンスチューニングガイド](https://github.com/motional/nuplan-devkit/blob/master/docs/performance_tuning_guide.md)を参照してください。

______________________________________________________________________

## Devkit構成
コードは以下のディレクトリに整理されています。

```
nuplan_devkit 
├── ci - 継続的インテグレーション用コード。一般ユーザーには不要。 
├── docs - リポジトリとデータセットのドキュメント。 
├── nuplan - 主なソースフォルダ。 
│ ├── common - "database"や"planning"で共有されるコード。 
│ ├── database - データセットとマップをロード・レンダリングするためのコアDevkit。 
│ ├── planning - シミュレーション、トレーニング、評価用のスタンドアロンプランニングフレームワーク。 
│ ├── submission - プランニングチャレンジ用の提出エンジン。 
│ └── cli - コマンドラインインターフェイスツール。
└── tutorials - インタラクティブなチュートリアル。
```

## 引用情報
nuPlanを参照する場合、以下の引用を使用してください:
```
@INPROCEEDINGS{nuplan, title={NuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles}, author={H. Caesar, J. Kabzan, K. Tan et al.,}, booktitle={CVPR ADP3 workshop}, year=2021 }
```