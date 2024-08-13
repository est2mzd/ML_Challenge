import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, : x.size(1), :]
        return x


class TransFuser(nn.Module):
    def __init__(self):
        super().__init__()

        # 画像ブランチ
        self.img_conv1 = nn.Conv2d(3, 72, kernel_size=7, stride=4, padding=2)  # [batch_size, 3, 160, 704] -> [batch_size, 72, 40, 176]
        self.img_conv2 = nn.Conv2d(72, 216, kernel_size=3, stride=2, padding=1) # [batch_size, 72, 40, 176] -> [batch_size, 216, 20, 88]
        self.img_conv3 = nn.Conv2d(216, 576, kernel_size=3, stride=2, padding=1) # [batch_size, 216, 20, 88] -> [batch_size, 576, 10, 44]
        self.img_conv4 = nn.Conv2d(576, 1512, kernel_size=3, stride=2, padding=1) # [batch_size, 576, 10, 44] -> [batch_size, 1512, 5, 22]

        # LiDARブランチ
        self.lidar_conv1 = nn.Conv2d(3, 72, kernel_size=4, stride=4, padding=0)  # [batch_size, 3, 256, 256] -> [batch_size, 72, 64, 64]
        self.lidar_conv2 = nn.Conv2d(72, 216, kernel_size=3, stride=2, padding=1) # [batch_size, 72, 64, 64] -> [batch_size, 216, 32, 32]
        self.lidar_conv3 = nn.Conv2d(216, 576, kernel_size=3, stride=2, padding=1) # [batch_size, 216, 32, 32] -> [batch_size, 576, 16, 16]
        self.lidar_conv4 = nn.Conv2d(576, 1512, kernel_size=3, stride=2, padding=1) # [batch_size, 576, 16, 16] -> [batch_size, 1512, 8, 8]

        # Positional Encoding
        self.positional_encoding1 = PositionalEncoding(d_model=72)
        self.positional_encoding2 = PositionalEncoding(d_model=216)
        self.positional_encoding3 = PositionalEncoding(d_model=576)
        self.positional_encoding4 = PositionalEncoding(d_model=1512)

        # Transformer Encoder Layers
        # batch_firstパラメータがTrueであると、バッチの次元が最初に来るようにテンソルが並び替えられます。
        # ドはまりポイントなので、要注意！！
        encoder_layers1 = TransformerEncoderLayer(d_model=72, nhead=8, batch_first=True)
        encoder_layers2 = TransformerEncoderLayer(d_model=216, nhead=8, batch_first=True)
        encoder_layers3 = TransformerEncoderLayer(d_model=576, nhead=8, batch_first=True)
        encoder_layers4 = TransformerEncoderLayer(d_model=1512, nhead=8, batch_first=True)

        # Transformer Encoders
        self.transformer1 = TransformerEncoder(encoder_layers1, num_layers=1)
        self.transformer2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.transformer3 = TransformerEncoder(encoder_layers3, num_layers=1)
        self.transformer4 = TransformerEncoder(encoder_layers4, num_layers=1)

        # Adaptive Average Pooling and Fully Connected Layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 512の青い箱 - Feature Reduction
        self.feature_reduction = nn.Linear(1512, 512)  # 1512 -> 512の全結合層
        
        # ピンクのMLP - Multi-Layer Perceptron
        self.mlp_fc1 = nn.Linear(512, 256)
        self.mlp_fc2 = nn.Linear(256, 128)

        # 64の青い箱 - Dimensionality Reduction
        self.dimensionality_reduction = nn.Linear(128, 64)  # 128 -> 64の全結合層

        # GRU for waypoint prediction (4 points)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.fc_gru = nn.Linear(64, 2)  # 2次元のウェイポイント (x, y)

        # 補助タスクのためのデコーダの追加
        # セマンティックセグメンテーション
        self.segmentation_decoder = nn.Sequential(
            nn.Conv2d(1512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 7, kernel_size=1)  # セマンティッククラスの数
        )

        # 深度予測
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(1512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)  # 深度チャネルは1つ
        )

        # HDマップ予測
        self.hd_map_decoder = nn.Sequential(
            nn.Conv2d(1512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=1)  # HDマップクラスの数（道路、レーンマーク、その他）
        )

        # 車両検出
        self.vehicle_detection_decoder = nn.Sequential(
            nn.Conv2d(1512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)  # 車両検出チャネル
        )

    def fusion_block(self, img_features, lidar_features, positional_encoding, transformer, img_conv, lidar_conv):
        # Positional Encodingの適用
        img_features_pos = positional_encoding(img_features.flatten(2).permute(2, 0, 1))  # [batch_size, channels, H*W] -> [H*W, batch_size, channels]
        lidar_features_pos = positional_encoding(lidar_features.flatten(2).permute(2, 0, 1))  # [batch_size, channels, H*W] -> [H*W, batch_size, channels]

        # 特徴の結合
        features = torch.cat((img_features_pos, lidar_features_pos), dim=0)  # [img_H*W + lidar_H*W, batch_size, channels]

        # Transformerによる特徴融合
        transformer_output = transformer(features)  # Self-Attention層の出力

        # 出力をImageとLiDARに分割
        img_size = img_features_pos.size(0)
        img_out = transformer_output[:img_size].permute(1, 2, 0).view(*img_features.shape)  # [H*W, batch_size, channels] -> [batch_size, channels, H, W]
        lidar_out = transformer_output[img_size:].permute(1, 2, 0).view(*lidar_features.shape)  # [H*W, batch_size, channels] -> [batch_size, channels, H, W]

        # Positional Encodingされていない元の特徴と足し算
        img_features = img_out + img_features  # [batch_size, channels, H, W]
        lidar_features = lidar_out + lidar_features  # [batch_size, channels, H, W]

        # デバッグ用のprint文
        if True:
            print(f"img_features size after fusion: {img_features.size()}")
            print(f"lidar_features size after fusion: {lidar_features.size()}")
            print("---------------------------------------------------------")

        return img_features, lidar_features

    def forward(self, image, lidar):
        # 画像ブランチとLiDARブランチの特徴抽出-1
        img_features1 = self.img_conv1(image)  # [batch_size, 3, 160, 704] -> [batch_size, 72, 40, 176]
        lidar_features1 = self.lidar_conv1(lidar)  # [batch_size, 3, 256, 256] -> [batch_size, 72, 64, 64]
        img_features1, lidar_features1 = self.fusion_block(img_features1, lidar_features1, self.positional_encoding1, self.transformer1, self.img_conv1, self.lidar_conv1)

        # 画像ブランチとLiDARブランチの特徴抽出-2
        img_features2 = self.img_conv2(img_features1)  # [batch_size, 72, 40, 176] -> [batch_size, 216, 20, 88]
        lidar_features2 = self.lidar_conv2(lidar_features1)  # [batch_size, 72, 64, 64] -> [batch_size, 216, 32, 32]
        img_features2, lidar_features2 = self.fusion_block(img_features2, lidar_features2, self.positional_encoding2, self.transformer2, self.img_conv2, self.lidar_conv2)

        # 画像ブランチとLiDARブランチの特徴抽出-3
        img_features3 = self.img_conv3(img_features2)  # [batch_size, 216, 20, 88] -> [batch_size, 576, 10, 44]
        lidar_features3 = self.lidar_conv3(lidar_features2)  # [batch_size, 216, 32, 32] -> [batch_size, 576, 16, 16]
        img_features3, lidar_features3 = self.fusion_block(img_features3, lidar_features3, self.positional_encoding3, self.transformer3, self.img_conv3, self.lidar_conv3)

        # 画像ブランチとLiDARブランチの特徴抽出-4
        img_features4 = self.img_conv4(img_features3)  # [batch_size, 576, 10, 44] -> [batch_size, 1512, 5, 22]
        lidar_features4 = self.lidar_conv4(lidar_features3)  # [batch_size, 576, 16, 16] -> [batch_size, 1512, 8, 8]
        img_features4, lidar_features4 = self.fusion_block(img_features4, lidar_features4, self.positional_encoding4, self.transformer4, self.img_conv4, self.lidar_conv4)

        # 特徴の平均プーリング
        img_features = self.avg_pool(img_features4)  # [batch_size, 1512, 5, 22] -> [batch_size, 1512, 1, 1]
        lidar_features = self.avg_pool(lidar_features4)  # [batch_size, 1512, 8, 8] -> [batch_size, 1512, 1, 1]

        # 理由: 平均プーリングは、特徴マップのサイズを小さくし、重要な情報を抽出するのに役立ちます。
        # ここでは、各特徴マップをサイズ[1, 1]に縮小し、次の全結合層に入力するために準備します。

        # 特徴の結合
        fused_features = img_features + lidar_features  # [batch_size, 1512, 1, 1]

        # 理由: 画像とLiDARの特徴を結合することで、両方のデータソースからの情報を統合し、より強力な表現を得ることができます。

        # 512の青い箱 - Feature Reduction
        fused_features = fused_features.view(-1, 1512)  # [batch_size, 1512]
        fused_features = self.feature_reduction(fused_features)  # [batch_size, 1512] -> [batch_size, 512]

        # 理由: 全結合層は、特徴の次元を削減し、次のMLP層に入力するための適切な形式に変換します。

        # ピンクのMLP - Multi-Layer Perceptron
        fused_features = F.relu(self.mlp_fc1(fused_features))  # [batch_size, 512] -> [batch_size, 256]
        fused_features = F.relu(self.mlp_fc2(fused_features))  # [batch_size, 256] -> [batch_size, 128]

        # 理由: MLPは、特徴の高次元表現を学習し、次のGRU層に適した形式に変換します。

        # 64の青い箱 - Dimensionality Reduction
        fused_features = self.dimensionality_reduction(fused_features)  # [batch_size, 128] -> [batch_size, 64]

        # 理由: 最後の全結合層で次元を64に削減します。

        # GRUによるウェイポイント予測（4点）
        fused_features = fused_features.unsqueeze(1).repeat(1, 4, 1)  # [batch_size, 64] -> [batch_size, 4, 64] として4つの時系列データに拡張
        gru_output, _ = self.gru(fused_features)  # [batch_size, 4, 64]
        waypoints = self.fc_gru(gru_output)  # [batch_size, 4, 2]
        
        # 理由: GRUは、時系列データを処理するために使用され、4つのウェイポイントを予測します。各ウェイポイントは2次元の座標 (x, y) です。

        # 補助タスクの予測
        # セマンティックセグメンテーション
        segmentation_output = self.segmentation_decoder(img_features4)  # [batch_size, 1512, 5, 22] -> [batch_size, 7, 5, 22]

        # 深度予測
        depth_output = self.depth_decoder(img_features4)  # [batch_size, 1512, 5, 22] -> [batch_size, 1, 5, 22]

        # HDマップ予測
        hd_map_output = self.hd_map_decoder(lidar_features4)  # [batch_size, 1512, 8, 8] -> [batch_size, 3, 8, 8]

        # 車両検出
        vehicle_detection_output = self.vehicle_detection_decoder(lidar_features4)  # [batch_size, 1512, 8, 8] -> [batch_size, 1, 8, 8]

        return waypoints, segmentation_output, depth_output, hd_map_output, vehicle_detection_output

    def loss(self, predicted_waypoints, ground_truth_waypoints, seg_output, seg_target, depth_output, depth_target, hd_map_output, hd_map_target, vehicle_detection_output, vehicle_detection_target):
        # ウェイポイントの損失（平均二乗誤差）
        waypoint_loss_fn = nn.MSELoss()
        waypoint_loss = waypoint_loss_fn(predicted_waypoints, ground_truth_waypoints)
        
        # セマンティックセグメンテーションの損失（クロスエントロピー）
        segmentation_loss_fn = nn.CrossEntropyLoss()
        segmentation_loss = segmentation_loss_fn(seg_output, seg_target)

        # 深度予測の損失（平均絶対誤差）
        depth_loss_fn = nn.L1Loss()
        depth_loss = depth_loss_fn(depth_output, depth_target)

        # HDマップ予測の損失（クロスエントロピー）
        hd_map_loss_fn = nn.CrossEntropyLoss()
        hd_map_loss = hd_map_loss_fn(hd_map_output, hd_map_target)

        # 車両検出の損失（フォーカルロス）
        vehicle_detection_loss_fn = nn.BCEWithLogitsLoss()
        vehicle_detection_loss = vehicle_detection_loss_fn(vehicle_detection_output, vehicle_detection_target)

        # 総損失
        total_loss = waypoint_loss + segmentation_loss + depth_loss + hd_map_loss + vehicle_detection_loss
        return total_loss

# モデルの初期化
model = TransFuser()

# ダミー入力
image_input = torch.randn(8, 3, 160, 704)  # [batch_size, 3, 高さ, 幅]
lidar_input = torch.randn(8, 3, 256, 256)  # [batch_size, 3, 高さ, 幅]

# フォワードパスの実行
predicted_waypoints, segmentation_output, depth_output, hd_map_output, vehicle_detection_output = model(image_input, lidar_input)  # [batch_size, 4, 2]

# 出力の表示
print(predicted_waypoints.shape)  # 期待される出力: [8, 4, 2]

# ダミーのground_truthを生成
ground_truth_waypoints = torch.randn(8, 4, 2)  # [batch_size, 4, 2]
segmentation_target = torch.randint(0, 7, (8, 5, 22))  # セマンティックセグメンテーションのターゲット
depth_target = torch.randn(8, 1, 5, 22)  # 深度予測のターゲット
hd_map_target = torch.randint(0, 3, (8, 8, 8))  # HDマップのターゲット
vehicle_detection_target = torch.randint(0, 2, (8, 1, 8, 8)).float()  # 車両検出のターゲット

# 損失の計算
loss = model.loss(predicted_waypoints, ground_truth_waypoints, segmentation_output, segmentation_target, depth_output, depth_target, hd_map_output, hd_map_target, vehicle_detection_output, vehicle_detection_target)  # スカラー値

# 損失の表示
print(loss.item())  # 損失値の表示
