import torch
import torch.nn as nn

# 定数の定義
D     = 128  # エンコーディングの次元数 (特徴ベクトルの次元数)
N_R   = 10   # 横方向クエリの数 (レーンや参照線の数)
N_L   = 5    # 縦方向クエリの数 (進行方向の経路の数)
L_enc = 4    # トランスフォーマー・エンコーダのレイヤー数
L_dec = 4    # トランスフォーマー・デコーダのレイヤー数
N_A   = 10   # 動的エージェントの数 (他の車両や歩行者など)
N_S   = 5    # 静的エージェントの数 (信号機、標識など)
N_P   = 20   # ポリラインの数 (道路やレーンの線形情報)
B     = 2    # バッチサイズ
S     = 15   # シーケンスの長さ
N_H   = 8    # アテンションヘッドの数

# エンコーダ部分の定義 (図の "Transformer Encoder" と対応、数式(4) のエンコード処理に相当)
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # PyTorchのnn.TransformerEncoder では、入力テンソルの形状は通常以下の順序で期待されます
        # src: (シーケンス長, バッチサイズ, 埋め込み次元)
        return self.transformer_encoder(src)

# デコーダ部分の定義 (図の "Lateral SelfAttn", "Longitudinal SelfAttn", "Query2Scene CrossAttn" に対応、数式(6)と(7)に相当)
class TrajectoryDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TrajectoryDecoder, self).__init__()
        self.lateral_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)  # 横方向の自己注意 (数式(6)の一部)
        self.longitudinal_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)  # 縦方向の自己注意 (数式(6)の一部)
        self.query_to_scene_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)  # クエリとシーンのクロスアテンション (数式(7))
        self.traj_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))  # 軌道を生成するMLP (数式(8)のT_0)
        self.score_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))  # スコアを生成するMLP (数式(8)のpi_0)

    def forward(self, Q_lat, Q_lon, E_enc):
        # Q_lat: (N_R, B, D), Q_lon: (N_L, B, D), E_enc: (S, B, D)
        Q_lat = Q_lat.permute(1, 0, 2)  # 軸の入れ替え: (B, N_R, D)
        Q_lon = Q_lon.permute(1, 0, 2)  # 軸の入れ替え: (B, N_L, D)
        Q_lat, _ = self.lateral_self_attn(Q_lat, Q_lat, Q_lat)  # 横方向の自己注意を適用
        Q_lon, _ = self.longitudinal_self_attn(Q_lon, Q_lon, Q_lon)  # 縦方向の自己注意を適用
        
        # 次元を合わせるため、E_enc のサイズを調整
        E_enc = E_enc.permute(1, 0, 2)  # (B, S, D) に変換
        Q_combined, _ = self.query_to_scene_cross_attn(Q_lat, E_enc, E_enc)  # クエリとシーンのクロスアテンションを適用
        
        T_0 = self.traj_mlp(Q_combined)  # 軌道を生成
        pi_0 = self.score_mlp(Q_combined)  # スコアを生成
        return T_0, pi_0

# モデルのインスタンス化 (図の "Transformer Encoder", "Lateral SelfAttn", "Logitudinal SelfAttn", "Query2Scene CrossAttn" に対応)
encoder = TransformerEncoder(d_model=D, nhead=N_H, num_layers=L_enc)
decoder = TrajectoryDecoder(d_model=D, nhead=N_H, num_layers=L_dec)

# ダミーデータの生成 (図の "Polyline Encoder" から得られる Q_lat と Q_lon に相当)
Q_lat = torch.randn(N_R, B, D)  # 横方向のクエリ
Q_lon = torch.randn(N_L, B, D)  # 縦方向のクエリ
src = torch.randn(S, B, D)  # エンコーダに入力するシーケンスデータ (図の "E_A", "E_O", "E_P", "E_AV" をまとめたもの)

# エンコーダの実行 (数式(4)に相当)
E_enc = encoder(src)

# デコーダの実行 (数式(6)から(8)に相当)
T_0, pi_0 = decoder(Q_lat, Q_lon, E_enc)

# 出力を確認
T_0, pi_0

