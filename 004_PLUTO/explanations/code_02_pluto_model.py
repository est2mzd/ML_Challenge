from copy import deepcopy  # deepcopyをインポート
import math  # 数学関数をインポート

import torch  # PyTorchをインポート
import torch.nn as nn  # ニューラルネットワークモジュールをインポート
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # TrajectorySamplingをインポート
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper  # TorchModuleWrapperをインポート
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)  # EgoTrajectoryTargetBuilderをインポート

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder  # PlutoFeatureBuilderをインポート

from .layers.fourier_embedding import FourierEmbedding  # FourierEmbeddingをインポート
from .layers.transformer import TransformerEncoderLayer  # TransformerEncoderLayerをインポート
from .modules.agent_encoder import AgentEncoder  # AgentEncoderをインポート
from .modules.agent_predictor import AgentPredictor  # AgentPredictorをインポート
from .modules.map_encoder import MapEncoder  # MapEncoderをインポート
from .modules.static_objects_encoder import StaticObjectsEncoder  # StaticObjectsEncoderをインポート
from .modules.planning_decoder import PlanningDecoder  # PlanningDecoderをインポート
from .layers.mlp_layer import MLPLayer  # MLPLayerをインポート

# 無意味な初期化だが、nuplanの要件で必要
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)  # TrajectorySamplingの初期化 (Fig1: Trajectory Sampling, Eq: 1)

class PlanningModel(TorchModuleWrapper):  # PlanningModelクラスの定義（TorchModuleWrapperを継承） (Fig1: Planning Model)
    def __init__(  # 初期化メソッド (Fig1: Initialization, Eq: 2)
        self,
        dim=128,  # 特徴次元
        state_channel=6,  # 状態チャネル数
        polygon_channel=6,  # ポリゴンチャネル数
        history_channel=9,  # 履歴チャネル数
        history_steps=21,  # 履歴ステップ数
        future_steps=80,  # 未来ステップ数
        encoder_depth=4,  # エンコーダーの深さ
        decoder_depth=4,  # デコーダーの深さ
        drop_path=0.2,  # ドロップパス率
        dropout=0.1,  # ドロップアウト率
        num_heads=8,  # 注意機構のヘッド数
        num_modes=6,  # モード数
        use_ego_history=False,  # 自己履歴の使用フラグ
        state_attn_encoder=True,  # 状態アテンションエンコーダーの使用フラグ
        state_dropout=0.75,  # 状態ドロップアウト率
        use_hidden_proj=False,  # 隠れ層プロジェクションの使用フラグ
        cat_x=False,  # 入力結合のフラグ
        ref_free_traj=False,  # 参照線なしの軌跡生成フラグ
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),  # FeatureBuilderの初期化
    ) -> None:
        super().__init__(  # 親クラスの初期化 (Fig1: TorchModuleWrapper Initialization)
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],  # 目標ビルダーの初期化 (Fig1: Target Builder, Eq: 3)
            future_trajectory_sampling=trajectory_sampling,  # 将来の軌跡サンプリングの設定
        )

        self.dim = dim  # 特徴次元の保存
        self.history_steps = history_steps  # 履歴ステップ数の保存
        self.future_steps = future_steps  # 未来ステップ数の保存
        self.use_hidden_proj = use_hidden_proj  # 隠れ層プロジェクション使用フラグの保存
        self.num_modes = num_modes  # モード数の保存
        self.radius = feature_builder.radius  # FeatureBuilderからの半径の取得
        self.ref_free_traj = ref_free_traj  # 参照線なし軌跡生成フラグの保存

        self.pos_emb = FourierEmbedding(3, dim, 64)  # 位置エンコーディングの初期化 (Fig1: Position Embedding, Eq: 4)

        self.agent_encoder = AgentEncoder(  # エージェントエンコーダーの初期化 (Fig1: Agent Encoder, Eq: 5)
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(  # 地図エンコーダーの初期化 (Fig1: Map Encoder, Eq: 6)
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)  # 静的物体エンコーダーの初期化 (Fig1: Static Objects Encoder, Eq: 7)

        self.encoder_blocks = nn.ModuleList(  # エンコーダーブロックの初期化 (Fig1: Transformer Encoder, Eq: 8)
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)  # 正規化レイヤーの初期化 (Fig1: Layer Normalization, Eq: 9)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)  # エージェント予測器の初期化 (Fig1: Agent Predictor, Eq: 10)
        self.planning_decoder = PlanningDecoder(  # プランニングデコーダーの初期化 (Fig1: Planning Decoder, Eq: 11)
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

        if use_hidden_proj:  # 隠れ層プロジェクションが使用される場合 (Fig1: Hidden Projection, Eq: 12)
            self.hidden_proj = nn.Sequential(  # 隠れ層プロジェクションの初期化
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if self.ref_free_traj:  # 参照線なし軌跡生成が使用される場合 (Fig1: Ref-Free Trajectory Decoder, Eq: 13)
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)  # 参照線なしデコーダーの初期化

        self.apply(self._init_weights)  # 重み初期化関数の適用 (Fig1: Weight Initialization, Eq: 14)

    def _init_weights(self, m):  # 重み初期化関数 (Eq: 14)
        if isinstance(m, nn.Linear):  # 線形層の場合
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier初期化 (Eq: 14a)
            if isinstance(m, nn.Linear) and m.bias is not None:  # バイアスが存在する場合
                nn.init.constant_(m.bias, 0)  # バイアスを0で初期化 (Eq: 14b)
        elif isinstance(m, nn.LayerNorm):  # レイヤーノルムの場合
            nn.init.constant_(m.bias, 0)  # バイアスを0で初期化 (Eq: 14c)
            nn.init.constant_(m.weight, 1.0)  # 重みを1で初期化 (Eq: 14d)
        elif isinstance(m, nn.BatchNorm1d):  # バッチノルムの場合
            nn.init.ones_(m.weight)  # 重みを1で初期化 (Eq: 14e)
            nn.init.zeros_(m.bias)  # バイアスを0で初期化 (Eq: 14f)
        elif isinstance(m, nn.Embedding):  # 埋め込み層の場合
            nn.init.normal_(m.weight, mean=0.0, std=0.02)  # 正規分布で重みを初期化 (Eq: 14g)

    def forward(self, data):  # フォワード関数 (Fig1: Forward Pass, Eq: 15)
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]  # エージェントの位置 (Fig1: Input Processing, Eq: 15a)
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]  # エージェントの向き (Fig1: Input Processing, Eq: 15b)
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]  # エージェントの有効マスク (Fig1: Input Processing, Eq: 15c)
        polygon_center = data["map"]["polygon_center"]  # ポリゴンの中心 (Fig1: Input Processing, Eq: 15d)
        polygon_mask = data["map"]["valid_mask"]  # ポリゴンの有効マスク (Fig1: Input Processing, Eq: 15e)

        bs, A = agent_pos.shape[0:2]  # バッチサイズとエージェント数 (Fig1: Batch Processing, Eq: 15f)

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)  # 位置の結合 (Fig1: Feature Concatenation, Eq: 16)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)  # 角度の結合 (Fig1: Feature Concatenation, Eq: 17)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi  # 角度の正規化 (Fig1: Angle Normalization, Eq: 18)
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)  # 位置と角度の結合 (Fig1: Combined Features, Eq: 19)

        agent_key_padding = ~(agent_mask.any(-1))  # エージェントのキー・パディング・マスク (Fig1: Padding Mask, Eq: 20a)
        polygon_key_padding = ~(polygon_mask.any(-1))  # ポリゴンのキー・パディング・マスク (Fig1: Padding Mask, Eq: 20b)
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)  # キー・パディング・マスクの結合 (Fig1: Combined Padding Mask, Eq: 20c)

        x_agent = self.agent_encoder(data)  # エージェントエンコーダーの実行 (Fig1: Agent Encoder, Eq: 5)
        x_polygon = self.map_encoder(data)  # マップエンコーダーの実行 (Fig1: Map Encoder, Eq: 6)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)  # 静的物体エンコーダーの実行 (Fig1: Static Objects Encoder, Eq: 7)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)  # エンコードされた特徴の結合 (Fig1: Feature Aggregation, Eq: 21)

        pos = torch.cat([pos, static_pos], dim=1)  # 位置の結合 (Fig1: Position Aggregation, Eq: 22)
        pos_embed = self.pos_emb(pos)  # 位置エンベディング (Fig1: Position Embedding, Eq: 4)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)  # キー・パディング・マスクの結合 (Fig1: Combined Padding Mask, Eq: 20c)
        x = x + pos_embed  # エンコードされた特徴に位置エンベディングを加算 (Fig1: Add Position Embedding, Eq: 23)

        for blk in self.encoder_blocks:  # エンコーダーブロックのループ (Fig1: Transformer Encoder, Eq: 8)
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)  # ブロックの実行 (Fig1: Transformer Block, Eq: 8)
        x = self.norm(x)  # 正規化 (Fig1: Layer Normalization, Eq: 9)

        prediction = self.agent_predictor(x[:, 1:A])  # エージェント予測の実行 (Fig1: Agent Predictor, Eq: 10)

        ref_line_available = data["reference_line"]["position"].shape[1] > 0  # 参照線が利用可能か確認 (Fig1: Reference Line Check, Eq: 24)

        if ref_line_available:  # 参照線が利用可能な場合
            trajectory, probability = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
            )  # プランニングデコーダーの実行 (Fig1: Planning Decoder, Eq: 11)
        else:
            trajectory, probability = None, None  # 参照線がない場合はNone (Fig1: No Trajectory, Eq: 25)

        out = {
            "trajectory": trajectory,  # 軌跡 (Fig1: Output Trajectory, Eq: 26a)
            "probability": probability,  # 確率 (Fig1: Output Probability, Eq: 26b)
            "prediction": prediction,  # 予測結果 (Fig1: Output Prediction, Eq: 26c)
        }

        if self.use_hidden_proj:  # 隠れ層プロジェクションが使用されている場合
            out["hidden"] = self.hidden_proj(x[:, 0])  # 隠れ層の出力を保存 (Fig1: Hidden Projection Output, Eq: 27)

        if self.ref_free_traj:  # 参照線なしの軌跡生成が使用されている場合
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )  # 参照線なし軌跡のデコード (Fig1: Ref-Free Trajectory Decoder, Eq: 13)
            out["ref_free_trajectory"] = ref_free_traj  # 参照線なし軌跡を保存 (Fig1: Output Ref-Free Trajectory, Eq: 28)

        if not self.training:  # モデルが訓練中でない場合
            if self.ref_free_traj:  # 参照線なしの軌跡生成が使用されている場合
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )  # 参照線なし軌跡の角度計算 (Fig1: Ref-Free Trajectory Angle, Eq: 29)
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )  # 参照線なし軌跡と角度の結合 (Fig1: Ref-Free Trajectory Processing, Eq: 30)
                out["output_ref_free_trajectory"] = ref_free_traj  # 参照線なし軌跡を出力 (Fig1: Final Ref-Free Trajectory, Eq: 31)

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],  # 位置予測の調整 (Fig1: Prediction Adjustment, Eq: 32a)
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],  # 角度予測の調整 (Fig1: Prediction Adjustment, Eq: 32b)
                    prediction[..., 4:6],  # その他の予測情報
                ],
                dim=-1,
            )  # 出力予測の結合 (Fig1: Final Prediction Output, Eq: 32)
            out["output_prediction"] = output_prediction  # 出力予測を保存 (Fig1: Output Prediction, Eq: 32c)

            if trajectory is not None:  # 軌跡が存在する場合
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)  # 参照線のパディングマスク (Fig1: Reference Line Padding, Eq: 33a)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)  # パディングされた部分にマスクを適用 (Fig1: Probability Masking, Eq: 33b)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])  # 軌跡の角度計算 (Fig1: Trajectory Angle Calculation, Eq: 34)
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )  # 軌跡と角度の結合 (Fig1: Trajectory Processing, Eq: 35)

                bs, R, M, T, _ = out_trajectory.shape  # 形状の取得 (Fig1: Shape Extraction, Eq: 36)
                flattened_probability = probability.reshape(bs, R * M)  # 確率の平坦化 (Fig1: Probability Flattening, Eq: 37)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]  # 最適な軌跡を選択 (Fig1: Best Trajectory Selection, Eq: 38)

                out["output_trajectory"] = best_trajectory  # 最適な軌跡を保存 (Fig1: Output Best Trajectory, Eq: 39)
                out["candidate_trajectories"] = out_trajectory  # 候補軌跡を保存 (Fig1: Output Candidate Trajectories, Eq: 40)
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]  # 参照線なしの最適軌跡 (Fig1: Output Trajectory without Reference, Eq: 41)
                out["probability"] = torch.zeros(1, 0, 0)  # 確率をゼロで初期化 (Fig1: Probability Zero Initialization, Eq: 42)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )  # 候補軌跡をゼロで初期化 (Fig1: Candidate Trajectories Zero Initialization, Eq: 43)

        return out  # 出力を返す (Fig1: Final Output, Eq: 44)
