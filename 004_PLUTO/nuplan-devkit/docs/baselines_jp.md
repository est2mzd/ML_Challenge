# ベースライン

開発キットにはいくつかのベースラインが用意されています。これらのベースラインは、新しいプランナーを比較する際の標準的な基準点となります。また、ベースラインは、ユーザーが自身のプランナーを試作したり、調整したりする際の出発点としても役立ちます。

## SimplePlanner

SimplePlannerはその名前が示す通り、計画能力がほとんどありません。このプランナーは、一定の速度で直線を計画します。このプランナーの唯一のロジックは、現在の速度が`max_velocity`を超えた場合に減速することです。

コードへのリンク: [SimplePlannerのコード](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/planner/simple_planner.py)

## IDMPlanner

Intelligent Driver Model Planner（IDMPlanner）は、以下の2つのパートで構成されています。

1. 経路計画
2. 縦方向制御（IDMポリシー）

### 経路計画

経路計画には幅優先探索アルゴリズムが使用されます。このアルゴリズムは、ミッションのゴールに向かう経路を見つけます。この経路は、一連の車線や車線接続から構成されており、ゴールを含む道路区画に導きます。見つかった経路からベースラインが抽出され、プランナーの参照経路として使用されます。

### IDMポリシー

プランナーが参照経路を持った後、その経路に沿ってどれだけ速く進むべきかを決定する必要があります。このために[IDMポリシー](https://en.wikipedia.org/wiki/Intelligent_driver_model)を使用します。このポリシーは、プランナーが他のエージェントとの距離に基づいてどの程度の速度で進むべきかを記述します。もちろん、プランナーの経路上にいる最も近いエージェントを選ぶのが賢明です。

したがって、IDMPlannerは幅優先探索を使用してミッションのゴールに向かう経路を見つけ、IDMポリシーによってその経路をどの程度進むべきかを決定します。

コードへのリンク: [IDMPlannerのコード](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/planner/idm_planner.py)

## UrbanDriverOpenLoopModel（MLPlanner）

UrbanDriverOpenLoopModelは、`MLPlanner`インターフェースを使用した機械学習プランナーの例として機能します。この実装は、L5Kitの["Urban Driver: Learning to Drive from Real-world Demonstrations Using Policy Gradients"](https://woven-planet.github.io/l5kit/urban_driver.html)の[実装](https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py)をnuPlanフレームワークに適応させたオープンループバージョンです。

このモデルは、ベクトル化されたエージェントと地図の入力を処理し、局所的な特徴記述子を生成します。これらはグローバル注意メカニズムに渡され、予測される自車軌跡が生成されます。このモデルは、nuPlanデータセットに含まれる専門家の軌跡と一致するように模倣学習を用いて訓練されます。訓練中には、データ分布のドリフトを軽減するためにエージェントと専門家の軌跡にいくらかのデータ拡張が行われます。

コードへのリンク: [UrbanDriverOpenLoopModelのコード](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/training/modeling/models/urban_driver_open_loop_model.py)
