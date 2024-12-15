# 評価指標 (Metrics)

このドキュメントでは、シナリオの評価とプランナーのスコアリングに使用されるnuPlanの評価指標について説明します。

---

## Challenge 1の共通評価指標

Challenge 1では、エキスパートが運転した軌跡とプランナーが提案する軌跡が特定の頻度（`1Hz`）でサンプリングされ、選択された時間範囲（`[3, 5, 8]s`）内で比較されます。このチャレンジでは、提出プランナーは最小頻度`1Hz`と最小計画範囲`6s`を満たす必要があります。

- **平均変位誤差 (ADE)**  
  時間ごとのサンプリング時点で、提案されたプランナー軌跡とエキスパート軌跡の各点のL2距離の平均を計算します。すべてのサンプリング時点でADEを計算し、その平均をシナリオのスコアとして定義します。

- **最終変位誤差 (FDE)**  
  サンプリングされた時点で、プランナー軌跡とエキスパート軌跡の最終点間のL2距離を計算します。すべてのサンプリング時点でFDEを計算し、その平均をシナリオのスコアとして定義します。

- **ミス率**  
  サンプリングされた時点で、提案されたプランナー軌跡とエキスパート軌跡のL2距離が最大変位閾値を超える場合、その時点の軌跡を「ミス」とみなします。シナリオ内のすべてのサンプリング時点でのミス率を計算し、最大許容閾値（`0.3`）を超える場合は不合格となります。

- **平均方位誤差 (AHE)**  
  提案されたプランナー軌跡とエキスパート軌跡の方位差の絶対値の平均を計算します。すべてのサンプリング時点で計算し、その平均をシナリオのスコアとして定義します。

- **最終方位誤差 (FHE)**  
  提案されたプランナー軌跡とエキスパート軌跡の最終時点での方位差の絶対値を計算します。すべてのサンプリング時点で計算し、その平均をシナリオのスコアとして定義します。

| Metric | 閾値 | 可視化 |
|:-------|:-----|:-------|
| [ADE](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/planner_expert_average_l2_error_within_bound.py) | 比較範囲: [3,5,8]s<br>頻度: 1Hz<br>最大平均誤差: 8m | 平均ADEのヒストグラム |
| [FDE](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/planner_expert_final_l2_error_within_bound.py) | 比較範囲: [3,5,8]s<br>頻度: 1Hz<br>最大最終誤差: 8m | 平均FDEのヒストグラム |
| [ミス率](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/planner_miss_rate_within_bound.py) | 比較範囲: [3,5,8]s<br>頻度: 1Hz<br>最大ミス率: 0.3 | ミス率のヒストグラム |
| [AHE](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/average_heading_error_within_bound.py) | 比較範囲: [3,5,8]s<br>頻度: 1Hz<br>最大方位誤差: 0.8rad | 平均方位誤差のヒストグラム |
| [FHE](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/final_heading_error_within_bound.py) | 比較範囲: [3,5,8]s<br>頻度: 1Hz<br>最大最終方位誤差: 0.8rad | 最終方位誤差のヒストグラム |

---

## Challenge 2とChallenge 3の共通評価指標

Challenge 2および3では、エキスパートの軌跡と自車（Ego）の軌跡を比較してシナリオを評価します。

- **責任のある衝突なし**  
  衝突とは、自車の境界ボックスと他のエージェントの境界ボックスが交差するイベントを指します。特定の衝突が複数フレームにわたる場合でも、最初のフレームのみを考慮します。適切な計画があれば回避できた衝突（責任ある衝突）のみが評価に含まれます。

- **走行可能領域の遵守**  
  自車は常にマップで指定された走行可能領域内を走行する必要があります。指定された閾値（`0.3m`）を超えて走行可能領域外に侵入する場合、このスコアは0となります。

- **進行方向の遵守**  
  自車が逆走するとペナルティが課されます。基準となる車線に対して移動方向を評価し、逆走距離に基づいてスコアが設定されます。

- **進行の達成**  
  エキスパートのルートに沿った自車の進行率を評価します。進行率が閾値（`0.2`）以上の場合、スコアは1となります。

- **衝突までの時間 (TTC)**  
  現在の速度と方位で進行した場合に自車と他のトラックが衝突するまでの時間を計算します。指定された時間範囲内で最初に交差する時点をTTCとして定義します。

- **速度制限の遵守**  
  自車の速度が地図で指定された速度制限を超える場合、違反としてカウントされます。違反の頻度と平均値が評価に使用されます。

- **快適性**  
  自車の軌跡における縦方向・横方向加速度、ヨー角速度、ジャークなどを評価し、経験的に設定された閾値内であれば快適とみなされます。

| Metric | 閾値 | 可視化 |
|:-------|:-----|:-------|
| [責任ある衝突なし](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/no_ego_at_fault_collisions.py) | 衝突許容値: 車両=0, オブジェクト=1 | 責任ある衝突数のヒストグラム |
| [走行可能領域の遵守](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/drivable_area_compliance.py) | 最大違反距離: 0.3m | 違反なしの場合のブール値 |
| [進行方向の遵守](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/driving_direction_compliance.py) | 逆走距離閾値: 6m | スコア値のヒストグラム |
| [進行の達成](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/ego_is_making_progress.py) | 最小進行閾値: 0.2 | ブール値ヒストグラム |
| [TTC](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/time_to_collision_within_bound.py) | 時間範囲: 3.0s | 最小TTCのヒストグラム |
| [速度制限の遵守](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/speed_limit_compliance.py) | 最大速度超過: 2.23m/s | 違反数のヒストグラム |
| [快適性](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/ego_is_comfortable.py) | 縦加速度: -4.05〜2.40m/s^2 他 | ブール値ヒストグラム |
