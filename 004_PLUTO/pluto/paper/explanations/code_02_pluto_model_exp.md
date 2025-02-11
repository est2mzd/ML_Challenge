### PLUTOモデルのコードと論文の対応関係

このコードが実行している部分は、添付PDFの **Fig.1** で示された **PLUTOモデル** の主要なコンポーネントに対応しています。

#### コード内の対応箇所と論文の対応関係

1. **`AgentEncoder`** と **`MapEncoder`** および **`StaticObjectsEncoder`**:
   - **Fig.1** の中で、「EA (エージェントの埋め込み)」、「EO (静的障害物の埋め込み)」、「EP (地図の埋め込み)」に対応しています。
   - 論文の数式 (2) に対応しており、エンコードされた埋め込み $E_0$ を生成します。
   - 数式 (2): $E_0 = \text{concat}(E_{AV}, E_A, E_O, E_P) + PE + E_{\text{attr}}$

2. **`TransformerEncoderLayer`**:
   - **Fig.1** の「Transformer Encoder × L_enc」に対応しています。これは複数の Transformer エンコーダーレイヤーで、エンコードされたシーン情報をさらに統合します。
   - 数式 (3) に該当し、$E_{\text{enc}}$ として出力されます。
   - 数式 (3): $E_i = E_{i-1} + \text{MHA}(Ê_{i-1}, Ê_{i-1}, Ê_{i-1})$

3. **`PlanningDecoder`**:
   - **Fig.1** の「Trajectory MLP」および「Score MLP」に対応しています。最終的な軌跡とスコアを生成する部分です。
   - 数式 (6) に対応し、$T_0$ および $\pi_0$ を生成します。
   - 数式 (6): $T_0 = \text{MLP}(Q_{\text{dec}}), \quad \pi_0 = \text{MLP}(Q_{\text{dec}})$

4. **`MLPLayer`** (if `ref_free_traj` is enabled):
   - 参照線なしの軌跡生成のためのデコーダーとして、数式 (7) に対応します。
   - 数式 (7): $\tau_{\text{free}} = \text{MLP}(E'_{AV})$
