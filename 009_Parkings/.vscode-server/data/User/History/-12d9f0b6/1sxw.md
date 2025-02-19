"""
# README: Optimization-based Motion Planning for Autonomous Parking

## 概要
本プロジェクトは、論文 **"Optimization-based Motion Planning for Autonomous Parking Considering Dynamic Obstacle: A Hierarchical Framework"** に基づき、
動的障害物を考慮した自律駐車手法を実装したものです。

本手法は **階層型フレームワーク** を採用し、
1. **高レベル経路計画 (SHA*)** : Scenario-based Hybrid A* により初期経路を生成
2. **低レベル軌道最適化 (NMPC)** : 非線形モデル予測制御を用いて最適軌道を生成

これにより、従来の Hybrid A* よりも計算効率を向上させつつ、動的障害物を考慮した駐車動作を実現しました。

---

## 1. 解決したい課題
### 主な課題:
- **駐車環境の制約** : 狭いスペースや障害物の影響
- **動的障害物の回避** : 移動する車両や歩行者を考慮
- **計算効率の向上** : 既存手法 (A*, Hybrid A*, MPC) は計算コストが高い

### 従来手法の問題点:
- **Hybrid A*** : 非ホロノミック制約に対応できるが、計算コストが高く局所最適解に陥りやすい
- **MPCベース手法** : 制約は多く扱えるが、リアルタイム適用が困難

本手法では、SHA*とNMPCを統合し、計算コストを抑えつつ、動的環境での駐車を可能にします。

---

## 2. 解決手法と結果
### **解決手法**
- **SHA* (Scenario-based Hybrid A*)** を用いて駐車経路を探索
- **NMPC (Nonlinear Model Predictive Control)** を用いて最適軌道を計算

### **実験結果**
- **148種類の駐車シナリオ** で評価
- **SHA* は Hybrid A* に比べて計算時間を約77%短縮**
- **駐車成功率 100%**

---

## 3. モデル構造
**キネマティック自転車モデル (Kinematic Bicycle Model)** を使用

```math
\begin{aligned}
    x_{k+1} &= x_k + T v_k \cos(\phi_k) \\
    y_{k+1} &= y_k + T v_k \sin(\phi_k) \\
    \phi_{k+1} &= \phi_k + T \frac{v_k}{L} \tan(\delta_k) \\
    v_{k+1} &= v_k + T a_k
\end{aligned}
```

- **状態変数:** $X_k = [x_k, y_k, \phi_k, v_k]$
- **制御入力:** $U_k = [\delta_k, a_k]$

---

## 4. 使用データセット
- **シミュレーション環境** : MATLAB 2021
- **最適化ソルバー** : IPOPT, CasADi, YALMIP
- **テストケース** : 148種類の異なる初期位置

---

## 5. 依存ライブラリ
Pythonの以下のライブラリが必要です。

```bash
pip install numpy casadi matplotlib imageio
```

---

## 6. 実行方法
Pythonスクリプトを実行してください。

```bash
python main.py
```

駐車動作の結果が **GIFファイル** として出力されます。

---

## 7. ライセンス
本プロジェクトは MIT ライセンスのもと公開されています。
"""
