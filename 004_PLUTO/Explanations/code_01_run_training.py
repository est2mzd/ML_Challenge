import logging
from typing import Optional

import hydra
import numpy
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from omegaconf import DictConfig

from src.custom_training import (
    TrainingEngine,
    build_training_engine,
    update_config_for_training,
)

# numba ライブラリのロギングレベルを警告のみ表示に設定
logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 環境変数によりデフォルトのデータセットと実験パスを上書きする設定
set_default_path()

# Hydra の設定ファイルのパスと名前を指定
CONFIG_PATH = "./config"  # Hydra の設定ファイルパス
CONFIG_NAME = "default_training"  # Hydra の設定ファイル名


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    トレーニングおよびバリデーションの実験のメインエントリポイント。
    :param cfg: omegaconf 辞書
    """
    # PyTorch Lightning のシードを設定
    pl.seed_everything(cfg.seed, workers=True)

    # ロガーの設定
    build_logger(cfg)

    # 設定を上書きし、設定内容を表示
    update_config_for_training(cfg)

    # 出力ストレージフォルダの作成
    build_training_experiment_folder(cfg=cfg)

    # ワーカーの構築
    worker = build_worker(cfg)

    # 設定に応じて異なる処理を実行
    if cfg.py_func == "train":
        # トレーニングエンジンの構築
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # トレーニングの開始
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(
                model=engine.model,  # モデル
                datamodule=engine.datamodule,  # データモジュール
                ckpt_path=cfg.checkpoint,  # チェックポイントパス
            )
        return engine

    elif cfg.py_func == "validate":
        # バリデーションエンジンの構築
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # バリデーションの開始
        logger.info("Starting validation...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
            engine.trainer.validate(
                model=engine.model,  # モデル
                datamodule=engine.datamodule,  # データモジュール
                ckpt_path=cfg.checkpoint,  # チェックポイントパス
            )
        return engine

    elif cfg.py_func == "test":
        # テストエンジンの構築
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # モデルのテスト
        logger.info("Starting testing...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(
                model=engine.model,  # モデル
                datamodule=engine.datamodule,  # データモジュール
            )
        return engine

    elif cfg.py_func == "cache":
        # 全特徴量を事前計算してキャッシュ
        logger.info("Starting caching...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None

    else:
        # 無効な関数名が指定された場合のエラー処理
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    main()
