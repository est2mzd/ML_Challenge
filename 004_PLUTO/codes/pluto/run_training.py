
#==========================================================================#
# 各種のimport設定
#==========================================================================#
import logging  # Python標準ライブラリのloggingモジュールをインポート。アプリケーションのログ管理に使用する。
from typing import Optional  # 型アノテーションのためのOptionalをインポート。戻り値がNoneになる可能性を扱えるようにする。

import hydra  # Hydraライブラリをインポート。動的設定管理と構成の切り替えに使用。
import numpy  # 数値計算ライブラリNumPyをインポート。数値演算や配列操作に使用。
import pytorch_lightning as pl  # PyTorch Lightningをインポート。PyTorchを使った機械学習モデルの簡易管理や学習プロセスの効率化を提供。

from nuplan.planning.script.builders.folder_builder import (  # nuPlanライブラリのフォルダ関連のビルダー関数をインポート。
    build_training_experiment_folder,  # トレーニング実験用のフォルダを構築するための関数。
)
from nuplan.planning.script.builders.logging_builder import build_logger  # ログ設定を構築する関数をインポート。
from nuplan.planning.script.builders.worker_pool_builder import build_worker  # ワーカープールを構築する関数をインポート。
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager  # プロファイリング用のコンテキストマネージャをインポート。
from nuplan.planning.script.utils import set_default_path  # デフォルトのパスを設定するためのユーティリティ関数をインポート。
from nuplan.planning.training.experiments.caching import cache_data  # データキャッシュ機能を提供する関数をインポート。
from omegaconf import DictConfig  # Hydra設定のための辞書型設定データ構造を提供するDictConfigをインポート。

from src.custom_training import (  # ユーザー定義のカスタムトレーニング関連モジュールをインポート。
    TrainingEngine,  # トレーニングエンジンの主要クラスをインポート。モデル学習の実行ロジックを含む。
    build_training_engine,  # トレーニングエンジンを構築する関数をインポート。
    update_config_for_training,  # トレーニング用に設定を更新する関数をインポート。
)


#==========================================================================#
# 準備
#==========================================================================#

# logger.info(), logger.warning()で、ログ出力ができるようになる
# ここでは、warning以上しか出さないようにしている
logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 環境変数がない場合デフォルト値を設定する : DEFAULT_DATA_ROOT, DEFAULT_EXP_ROOT
# ファイルパス = ./nuplan-devkit/nuplan/planning/script/utils.py
set_default_path()

# Hydraというライブラリで使用する config(yaml)のファイルパスを設定
# ここでは ./config/default_tarining.yaml　に設定されている情報を読み込みたい
CONFIG_PATH = "./config"
CONFIG_NAME = "default_training"

#==========================================================================#
# @hydra.main　の説明
# 1. hydra.main に config(yaml) のファイルパスを渡し、読み込む
# 2. config(yaml)の中身は "def main(cfg)"" の cfgの中に保存され、main() の中で使用される
# ** main関数の返り値の型 => Optional[TrainingEngine] => TrainingEngine or None が返り値の型
# ** DictConfig は、Hydraライブラリで提供される 設定データ構造 の一種で、Pythonの標準的な辞書型 (dict) を拡張したもの
#==========================================================================#
# Hydraデコレータを使用
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    トレーニング/検証実験のメインエントリポイント。
    :param cfg: omegaconf形式の設定データ
    """
    # PyTorch Lightningで再現性のための乱数シードを設定。`workers=True`で並列実行も含む。
    # yamlのなかに、 seed がない --> Hydraのデフォルト値が使用されている
    pl.seed_everything(cfg.seed, workers=True)

    # ロガーを構築し、設定に基づいたロギング環境を準備
    # ファイルパス = ./nuplan-devkit/nuplan/planning/script/builders/logging_builder.py
    build_logger(cfg)

    # 設定をトレーニング用にするために、下記を上書きする
    #   - cfg.cache.cache_path # キャッシュパスに基づいて、キャッシュディレクトリを削除・作成する動作が条件により実行
    #   - cfg.data_loader.params.num_workers # オーバーフィッティング検出が有効な場合に値を 0 に変更
    # ファイルパス = ./pluto/src/custom_training/custom_training_builder.py
    update_config_for_training(cfg)

    # トレーニングの出力データを保存するフォルダ(cfg.output_dir)を作成
    # config(yaml)では、output_dirが指定されていないので、Hydraのデフォルトパスが使われる
    # ファイルパス = ./nuplan-devkit/nuplan/planning/script/builders/folder_builder.py
    build_training_experiment_folder(cfg=cfg)

    # ワーカープール(データロードや並列処理用)を構築
    # ファイルパス = ./nuplan-devkit/nuplan/planning/script/builders/worker_pool_builder.py
    worker = build_worker(cfg)

    # 設定で指定されたpy_funcが "train" の場合、トレーニングを実行
    if cfg.py_func == "train":
        # トレーニングエンジンの構築をプロファイリングしながら実行
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker) # トレーニングエンジンの構築

        # トレーニングプロセスの開始をロギング
        logger.info("Starting training...")
        
        # トレーニングをプロファイリングしながら実行
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(                # トレーニングの実行: モデルとデータを使って実際に学習を行う。
                model=engine.model,            # トレーニング対象のモデル
                datamodule=engine.datamodule,  # データモジュール(データセットやローダーを管理)
                ckpt_path=cfg.checkpoint,      # 再開時のチェックポイントパス
            )
        return engine  # トレーニングエンジンを返す

    # 設定で指定されたpy_funcが "validate" の場合、検証を実行
    if cfg.py_func == "validate":
        # 検証エンジンの構築をプロファイリングしながら実行
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # 検証プロセスの開始をロギング
        logger.info("Starting validation...")
        
        # 検証をプロファイリングしながら実行
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
            engine.trainer.validate(
                model=engine.model,          # 検証対象のモデル
                datamodule=engine.datamodule,  # データモジュール
                ckpt_path=cfg.checkpoint,    # チェックポイントパス
            )
        return engine  # 検証エンジンを返す

    # 設定で指定されたpy_funcが "test" の場合、テストを実行
    elif cfg.py_func == "test":
        # テストエンジンの構築をプロファイリングしながら実行
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # テストプロセスの開始をロギング
        logger.info("Starting testing...")
        # テストをプロファイリングしながら実行
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine  # テストエンジンを返す

    # 設定で指定されたpy_funcが "cache" の場合、特徴量をキャッシュ
    elif cfg.py_func == "cache":
        # キャッシュプロセスの開始をロギング
        logger.info("Starting caching...")
        # 特徴量キャッシュ処理をプロファイリングしながら実行
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None  # キャッシュ処理には戻り値がない

    # 設定されたpy_funcが不正な場合、エラーをスロー
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


# スクリプトが直接実行された場合にmain関数を呼び出す
if __name__ == "__main__":
    main()
