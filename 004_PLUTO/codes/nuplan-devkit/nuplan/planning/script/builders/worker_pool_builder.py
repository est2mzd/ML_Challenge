import logging  # ロギング機能を提供する標準ライブラリ

from hydra.utils import instantiate  # Hydraのユーティリティ関数。設定を基にオブジェクトを動的にインスタンス化
from omegaconf import DictConfig  # YAMLや辞書形式の設定を扱うためのクラス

from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type  # 型チェックと検証に関する関数

# WorkerPool
#  - 並列処理やマルチスレッド処理の基礎となるクラス。
#  - タスクをワーカーに割り当て、処理結果を収集する機能を持つ。
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool  # ワーカープールの基底クラス

# RayDistributed:
#  - Ray ライブラリを使用した分散処理用のクラス。
#  - 高度な並列処理や大規模分散システム向けに設計されている。
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed  # Rayを使用した分散型ワーカークラス

logger = logging.getLogger(__name__)  # 現在のモジュール用のロガーを取得

def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    ワーカープールを構築する関数。
    :param cfg: DictConfig形式の設定。実験実行に使用される構成。
    :return: WorkerPoolのインスタンス。
    """
    logger.info('Building WorkerPool...')  # ワーカープール構築の開始をログ出力

    # ワーカープールをインスタンス化。RayDistributedかどうかで条件分岐
    worker: WorkerPool = (
        # instantiate = Hydra のユーティリティ関数。設定ファイル (cfg) に記述された情報を基に動的にクラスやオブジェクトを生成する。
        # cfg.worker     --> 定義場所 = ./nuplan-devkit/nuplan/planning/script/config/common/default_common.yaml
        # cfg.output_dir --> 定義場所 = ./nuplan-devkit/nuplan/planning/script/config/common/default_experiment.yaml
        # 下の3行は三項演算子を使った書き方
        # if is_target_type(cfg.worker, RayDistributed)
        #    True  --> instantiate(cfg.worker, output_dir=cfg.output_dir)
        #    False --> instantiate(cfg.worker)
        instantiate(cfg.worker, output_dir=cfg.output_dir)  # RayDistributed型の場合、出力ディレクトリを指定
        if is_target_type(cfg.worker, RayDistributed)  # cfg.workerがRayDistributed型かどうか判定
        else instantiate(cfg.worker)  # それ以外の場合、通常の方法でインスタンス化
    )
    
    # ファイルパス = ./nuplan-devkit/nuplan/planning/script/builders/utils/utils_type.py
    validate_type(worker, WorkerPool)  # インスタンスがWorkerPool型かどうかを検証

    logger.info('Building WorkerPool...DONE!')  # ワーカープール構築の完了をログ出力
    return worker  # 構築したWorkerPoolのインスタンスを返す
