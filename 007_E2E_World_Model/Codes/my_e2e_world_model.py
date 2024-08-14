import torch
import torch.nn as nn
import torch.nn.functional as F

# 定数
NUM_VIEWS = 4  # ビューの数
LATENT_DIM = 128  # ラテント特徴の次元
WAYPOINT_DIM = 64  # ウェイポイントの次元
SEQ_LEN = 10  # ウェイポイントシーケンスの長さ

# CrossAttentionモジュール (数式: v_i = CrossAttention(q_i, f_i, f_i))
class CrossAttention(nn.Module):
    def __init__(self, latent_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(latent_dim, latent_dim)  # クエリの線形変換
        self.key = nn.Linear(latent_dim, latent_dim)    # キーの線形変換
        self.value = nn.Linear(latent_dim, latent_dim)  # バリューの線形変換

    def forward(self, Q, K, V):
        # Q: クエリ [B, N, D]
        # K: キー [B, N, D]
        # V: バリュー [B, N, D]
        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)), dim=-1)  # アテンション重みを計算 [B, N, N]
        output = torch.bmm(attn_weights, V)  # アテンション重みに基づいてバリューを加重平均 [B, N, D]
        return output

# WaypointDecoderモジュール (ブロック: Waypoint Decoder, 数式: w_j = MLP(CrossAttention(q_j^{wp}, E, E)))
class WaypointDecoder(nn.Module):
    def __init__(self, latent_dim, waypoint_dim, seq_len):
        super(WaypointDecoder, self).__init__()
        self.cross_attention = CrossAttention(latent_dim)  # ウェイポイントのクロスアテンション
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, waypoint_dim),  # MLPでラテント特徴からウェイポイントへ変換
            nn.ReLU(),
            nn.Linear(waypoint_dim, seq_len * 2)  # x, y 座標を出力
        )

    def forward(self, Q_wp, E):
        # Q_wp: ウェイポイントクエリ, E: 強化されたラテント
        context = self.cross_attention(Q_wp, E, E)  # クロスアテンションを計算
        waypoints = self.mlp(context)  # ウェイポイントをデコード
        return waypoints.view(waypoints.size(0), SEQ_LEN, 2)  # [B, SEQ_LEN, 2]

# LatentWorldModelモジュール (ブロック: Latent World Model, 数式: P_{t+1} = LatentWorldModel(A_t))
class LatentWorldModel(nn.Module):
    def __init__(self, latent_dim):
        super(LatentWorldModel, self).__init__()
        self.transformer_decoder = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=8)  # トランスフォーマーデコーダ
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),  # トランスフォーマー出力をMLPで処理
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)  # 次のフレームの予測ラテントを生成
        )

    def forward(self, A_t):
        # A_t: アクションベースのビューラテント
        latent_pred = self.transformer_decoder(A_t, A_t)  # トランスフォーマーデコーダを通して次フレームを予測
        return self.mlp(latent_pred)  # 最終的な予測ラテントを出力

# EndToEndPlannerモジュール (ブロック: End-to-End Planner)
class EndToEndPlanner(nn.Module):
    def __init__(self, num_views, latent_dim, waypoint_dim, seq_len):
        super(EndToEndPlanner, self).__init__()
        # 各ビューごとのエンコーダ (ブロック: Encoder, 数式: F = {f_1, f_2, ..., f_N})
        self.encoder = nn.ModuleList([nn.Linear(3*64*64, latent_dim) for _ in range(num_views)])
        self.view_attention = CrossAttention(latent_dim)  # ビューアテンションモジュール (ブロック: View Attention)
        self.temporal_aggregation = nn.Linear(latent_dim, latent_dim)  # 時間的集約モジュール (ブロック: Temporal Aggregation)
        self.waypoint_decoder = WaypointDecoder(latent_dim, waypoint_dim, seq_len)  # ウェイポイントデコーダ (ブロック: Waypoint Decoder)

    def forward(self, x, H=None):
        # x: 入力画像 [B, NUM_VIEWS, C, H, W] (ブロック: Input Image)
        B = x.size(0)
        # 各ビューごとに特徴抽出 (数式: F = {f_1, f_2, ..., f_N})
        F = torch.stack([self.encoder[i](x[:, i].view(B, -1)) for i in range(NUM_VIEWS)], dim=1)  # [B, N, D]
        Q_view = F  # クエリとして特徴量そのものを使用
        V = self.view_attention(Q_view, F, F)  # 観察ビューラテントを計算 (数式: v_i = CrossAttention(q_i, f_i, f_i))

        # 強化されたラテント E = V + H (ブロック: Temporal Aggregation)
        if H is not None:
            E = V + H  # 歴史的ビューラテントがある場合は加算
        else:
            E = V  # 初期フレームの場合は観察ビューラテントのみを使用

        # ウェイポイントクエリ (学習可能) をランダムに初期化
        Q_wp = torch.randn(B, NUM_VIEWS, LATENT_DIM)
        waypoints = self.waypoint_decoder(Q_wp, E)  # ウェイポイントを予測 (数式: w_j = MLP(CrossAttention(q_j^{wp}, E, E)))
        return waypoints, E  # 予測されたウェイポイントと強化されたラテントを出力

# モデルのインスタンス化
end_to_end_planner = EndToEndPlanner(NUM_VIEWS, LATENT_DIM, WAYPOINT_DIM, SEQ_LEN)
latent_world_model = LatentWorldModel(LATENT_DIM)

# サンプル入力 (ブロック: Input Image)
x = torch.randn(8, NUM_VIEWS, 3, 64, 64)  # バッチサイズ8, 4ビュー, 3チャンネル, 64x64画像

# エンドツーエンドプランナーを通した特徴抽出とウェイポイント予測 (ブロック: End-to-End Planner, Waypoint Decoder)
waypoints, E_t = end_to_end_planner(x)

# アクションベースのラテント生成 (数式: a_i^t = MLP([e_i^t, w_t]))
w_t = waypoints.view(8, -1)  # フラットなウェイポイントベクトル
A_t = torch.cat([E_t, w_t.unsqueeze(1).expand(-1, NUM_VIEWS, -1)], dim=-1)  # [B, N, D+SEQ_LEN*2]

# ラテントワールドモデルを通した次フレームの予測 (ブロック: Latent World Model, 数式: P_{t+1} = LatentWorldModel(A_t))
P_t1 = latent_world_model(A_t)

# 次のステップでは P_t1 を用いて新たな H を生成し、さらなる予測を行います。
