import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Positional Encodingクラスの定義
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # ポジショナルエンコーディングを初期化
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(1)

    def forward(self, x):
        length = x.size(0)
        return x + self.encoding[:length, :]

# TransFuserクラスの定義
class TransFuser(nn.Module):
    def __init__(self):
        super(TransFuser, self).__init__()

        # 画像ブランチの畳み込み層
        self.img_conv1 = nn.Conv2d(3, 72, kernel_size=3, stride=2, padding=1)
        self.img_conv2 = nn.Conv2d(72, 216, kernel_size=3, stride=2, padding=1)
        self.img_conv3 = nn.Conv2d(216, 576, kernel_size=3, stride=2, padding=1)
        self.img_conv4 = nn.Conv2d(576, 1512, kernel_size=3, stride=2, padding=1)
        
        # LiDARブランチの畳み込み層
        self.lidar_conv1 = nn.Conv2d(3, 72, kernel_size=3, stride=2, padding=1)
        self.lidar_conv2 = nn.Conv2d(72, 216, kernel_size=3, stride=2, padding=1)
        self.lidar_conv3 = nn.Conv2d(216, 576, kernel_size=3, stride=2, padding=1)
        self.lidar_conv4 = nn.Conv2d(576, 1512, kernel_size=3, stride=2, padding=1)
        
        # Positional Encoding
        self.positional_encoding1 = PositionalEncoding(d_model=72)
        self.positional_encoding2 = PositionalEncoding(d_model=216)
        self.positional_encoding3 = PositionalEncoding(d_model=576)
        self.positional_encoding4 = PositionalEncoding(d_model=1512)
        
        # Transformerエンコーダー
        encoder_layers1 = TransformerEncoderLayer(d_model=72, nhead=8)
        encoder_layers2 = TransformerEncoderLayer(d_model=216, nhead=8)
        encoder_layers3 = TransformerEncoderLayer(d_model=576, nhead=8)
        encoder_layers4 = TransformerEncoderLayer(d_model=1512, nhead=8)
        
        self.transformer1 = TransformerEncoder(encoder_layers1, num_layers=1)
        self.transformer2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.transformer3 = TransformerEncoder(encoder_layers3, num_layers=1)
        self.transformer4 = TransformerEncoder(encoder_layers4, num_layers=1)
        
        # 最終層
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1512, 512)
        
        # MLPとGRU
        self.mlp = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.linear = nn.Linear(64, 2)
    
    # 畳み込みとTransformerの処理を関数化
    def conv_and_transformer(self, img_feat, lidar_feat, img_conv, lidar_conv, positional_encoding, transformer):
        img_feat_flat = img_feat.flatten(2).permute(2, 0, 1)
        img_feat_flat = positional_encoding(img_feat_flat)

        lidar_feat_flat = lidar_feat.flatten(2).permute(2, 0, 1)
        lidar_feat_flat = positional_encoding(lidar_feat_flat)

        combined_feat = torch.cat((img_feat_flat, lidar_feat_flat), dim=0)
        fused_feat = transformer(combined_feat)
        fused_feat_img, fused_feat_lidar = torch.split(fused_feat, [img_feat_flat.size(0), lidar_feat_flat.size(0)], dim=0)
        fused_feat_img = fused_feat_img.permute(1, 2, 0).view(-1, img_feat.size(1), img_feat.size(2), img_feat.size(3))
        fused_feat_lidar = fused_feat_lidar.permute(1, 2, 0).view(-1, lidar_feat.size(1), lidar_feat.size(2), lidar_feat.size(3))

        img_feat = img_feat + fused_feat_img
        lidar_feat = lidar_feat + fused_feat_lidar

        img_feat = img_conv(img_feat)
        lidar_feat = lidar_conv(lidar_feat)

        return img_feat, lidar_feat

    def forward(self, img, lidar):
        # 画像とLiDARの畳み込みとTransformer処理
        img_feat1 = self.img_conv1(img)
        lidar_feat1 = self.lidar_conv1(lidar)
        img_feat2, lidar_feat2 = self.conv_and_transformer(img_feat1, lidar_feat1, self.img_conv2, self.lidar_conv2, self.positional_encoding2, self.transformer2)
        img_feat3, lidar_feat3 = self.conv_and_transformer(img_feat2, lidar_feat2, self.img_conv3, self.lidar_conv3, self.positional_encoding3, self.transformer3)
        img_feat4, lidar_feat4 = self.conv_and_transformer(img_feat3, lidar_feat3, self.img_conv4, self.lidar_conv4, self.positional_encoding4, self.transformer4)
        
        # 最終層の出力をプールし、全結合層を通す
        pooled_feat4 = self.avg_pool(img_feat4).view(img_feat4.size(0), -1)
        out4 = self.fc(pooled_feat4)

        # MLPとGRUによるウェイポイント予測
        mlp_out = self.mlp(out4)
        gru_out, _ = self.gru(mlp_out.unsqueeze(1))
        waypoints = self.linear(gru_out.squeeze(1))
        
        return waypoints

# 使用例
model = TransFuser()

# 入力の例（バッチサイズ, チャンネル, 高さ, 幅）
img = torch.randn(1, 3, 320, 704)
lidar = torch.randn(1, 3, 256, 256)

# フォワードパス
waypoints = model(img, lidar)
print(waypoints)