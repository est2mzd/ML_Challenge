import torch
import torch.nn as nn
import torch.nn.functional as F

# 定数の定義
D = 128  # 埋め込みの次元数
N_R = 10  # 参照線の数（横方向クエリの数）
N_L = 5  # 縦方向クエリの数
L_enc = 4  # トランスフォーマーエンコーダのレイヤー数
L_dec = 4  # トランスフォーマーデコーダのレイヤー数
N_A = 10  # エージェントの数
N_S = 5  # 静的障害物の数
N_P = 20  # ポリラインの数

# エンコーダ部分の定義（図中の "Transformer Encoder" に相当）
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # src: (N, S, E) where S is the sequence length and E is the embedding dimension
        return self.transformer_encoder(src)

# デコーダ部分の定義（図中の "Lateral SelfAttn", "Logitudinal SelfAttn", "Query2Scene CrossAttn" に相当）
class TrajectoryDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TrajectoryDecoder, self).__init__()
        self.lateral_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)  # Lateral SelfAttn
        self.longitudinal_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)  # Longitudinal SelfAttn
        self.query_to_scene_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)  # Query2Scene CrossAttn
        self.traj_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))  # Trajectory MLP
        self.score_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))  # Score MLP

    def forward(self, Q_lat, Q_lon, E_enc):
        # Q_lat: (N_R, B, D), Q_lon: (N_L, B, D), E_enc: (S, B, D)
        Q_lat = Q_lat.permute(1, 0, 2)  # (B, N_R, D)
        Q_lon = Q_lon.permute(1, 0, 2)  # (B, N_L, D)
        Q_lat, _ = self.lateral_self_attn(Q_lat, Q_lat, Q_lat)  # Lateral attention
        Q_lon, _ = self.longitudinal_self_attn(Q_lon, Q_lon, Q_lon)  # Longitudinal attention
        Q_combined = torch.cat([Q_lat, Q_lon], dim=1)  # Combine lateral and longitudinal queries
        Q_combined, _ = self.query_to_scene_cross_attn(Q_combined, E_enc, E_enc)  # Cross-attention
        T_0 = self.traj_mlp(Q_combined)  # Trajectory output
        pi_0 = self.score_mlp(Q_combined)  # Score output
        return T_0, pi_0

# モデルの定義（図全体のアーキテクチャに相当）
class PLUTOModel(nn.Module):
    def __init__(self):
        super(PLUTOModel, self).__init__()
        self.encoder = TransformerEncoder(d_model=D, nhead=8, num_layers=L_enc)
        self.decoder = TrajectoryDecoder(d_model=D, nhead=8, num_layers=L_dec)
        self.polyline_encoder = nn.Linear(8, D)  # Polyline feature to embedding

    def forward(self, E_A, E_O, E_P, E_AV):
        # E_A: (N_A, B, D), E_O: (N_S, B, D), E_P: (N_P, B, D), E_AV: (1, B, D)
        # 数式 (3): エージェントの履歴データをベクトル化して入力とする
        E_combined = torch.cat([E_AV, E_A, E_O, E_P], dim=0)  # (N_A + N_S + N_P + 1, B, D)
        
        # 数式 (5): シーン全体のエンコードを実行
        E_enc = self.encoder(E_combined)  # Transformer Encoder

        # 数式 (4): 横方向クエリの生成
        Q_lat = self.polyline_encoder(E_P)  # Polyline encoder (Lateral queries)
        
        # 縦方向クエリを学習可能パラメータとして生成
        Q_lon = nn.Parameter(torch.randn(N_L, E_combined.size(1), D))  # Learnable longitudinal queries
        
        # 数式 (7): 軌道とスコアの生成
        T_0, pi_0 = self.decoder(Q_lat, Q_lon, E_enc)  # Decode trajectories and scores
        return T_0, pi_0

# デバイス設定: GPUが利用可能ならGPUを、そうでなければCPUを使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルのインスタンス化
model = PLUTOModel().to(device)

# ダミーデータの生成
E_A = torch.randn(N_A, batch_size, D).to(device)  # エージェント埋め込み
E_O = torch.randn(N_S, batch_size, D).to(device)  # 静的障害物埋め込み
E_P = torch.randn(N_P, batch_size, 8).to(device)  # ポリライン埋め込み（エンコード前）
E_AV = torch.randn(1, batch_size, D).to(device)  # AVの埋め込み

# フォワードパス
T_0, pi_0 = model(E_A, E_O, E_P, E_AV)

# 出力のサイズ確認
print("Trajectory Output Size:", T_0.size())  # Trajectory Outputの期待サイズ: (N_R + N_L, batch_size, 2)
print("Score Output Size:", pi_0.size())      # Score Outputの期待サイズ: (N_R + N_L, batch_size, 1)

# 学習用の設定
criterion = nn.MSELoss()  # 損失関数（ここではMSEを使用）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 最適化手法

# 学習ループの設定
num_epochs = 10  # 学習エポック数

for epoch in range(num_epochs):
    model.train()  # モデルを学習モードに設定
    optimizer.zero_grad()  # 勾配を初期化

    # ダミーターゲットの生成（例えば、ゼロベクトルを使用）
    target_T_0 = torch.zeros_like(T_0).to(device)
    target_pi_0 = torch.zeros_like(pi_0).to(device)

    # 損失の計算
    loss_trajectory = criterion(T_0, target_T_0)
    loss_score = criterion(pi_0, target_pi_0)
    loss = loss_trajectory + loss_score

    # 逆伝播と最適化
    loss.backward()  # 勾配を計算
    optimizer.step()  # モデルパラメータを更新

    # 各エポックの損失を表示
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# テストモードへの切り替え
model.eval()
with torch.no_grad():
    T_0, pi_0 = model(E_A, E_O, E_P, E_AV)
    # テスト時の出力サイズを表示
    print("Test - Trajectory Output Size:", T_0.size())
    print("Test - Score Output Size:", pi_0.size())

#============================================================#
# プログラムにおける数式、図との対応

# 数式 (1) と (2):
# PLUTOModel.forward: 自動運転車の計画軌道 (T_0, π_0) とエージェントの予測 P_{1:NA} を生成します。
# f(A, O, M, C | φ) としてモデル全体が機能し、図全体の流れで対応。

# 数式 (3):
# E_combined: エージェント、障害物、マップ、AVの状態を結合し、履歴データをベクトル化。

# 数式 (4):
# Q_lat = self.polyline_encoder(E_P): ポリライン情報を使って横方向のクエリを生成。

# 数式 (5):
# E_enc = self.encoder(E_combined): シーン全体をエンコードし、複雑な相互作用を捉える。

# 数式 (6) と (7):
# Q_lat, Q_lon, Q_combined: 横方向および縦方向の自己注意を適用し、シーン情報とのクロスアテンションを実行。

# 数式 (8):
# T_0 = self.traj_mlp(Q_combined), pi_0 = self.score_mlp(Q_combined): 軌道とスコアを最終的に生成。
#============================================================#