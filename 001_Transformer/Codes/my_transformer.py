import torch
import torch.nn as nn
import math

# Transformer全体の定義
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Encoder層とDecoder層を定義
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            ), 
            num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            ), 
            num_layers=num_decoder_layers
        )
        
        # 入力埋め込み層
        self.src_embedding = nn.Embedding(10000, d_model) # 入力トークンの埋め込み (ボキャブラリーサイズ: 10000, 埋め込み次元: d_model)
        self.tgt_embedding = nn.Embedding(10000, d_model) # 出力トークンの埋め込み (ボキャブラリーサイズ: 10000, 埋め込み次元: d_model)
        
        # 位置エンコーディング
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        # 出力の最終線形層
        self.fc_out = nn.Linear(d_model, 10000) # 出力の次元数をボキャブラリーサイズに変換する線形層
        
    def forward(self, src, tgt):
        # src: 入力シーケンス (シーケンス長, バッチサイズ)
        # tgt: 出力シーケンス (シーケンス長, バッチサイズ)
        
        # 入力シーケンスの埋め込みと位置エンコーディングの加算
        src = self.src_embedding(src) * math.sqrt(d_model) # (シーケンス長, バッチサイズ, d_model)
        src = self.positional_encoding(src) # 位置エンコーディングを加算した埋め込み
        
        # 出力シーケンスの埋め込みと位置エンコーディングの加算
        tgt = self.tgt_embedding(tgt) * math.sqrt(d_model) # (シーケンス長, バッチサイズ, d_model)
        tgt = self.positional_encoding(tgt) # 位置エンコーディングを加算した埋め込み
        
        # Encoderの処理
        memory = self.encoder(src) # (シーケンス長, バッチサイズ, d_model)
        
        # Decoderの処理
        output = self.decoder(tgt, memory) # (シーケンス長, バッチサイズ, d_model)
        
        # 出力の最終線形層
        output = self.fc_out(output) # (シーケンス長, バッチサイズ, 10000) 出力のボキャブラリーサイズに変換
        
        return output

# 位置エンコーディングを定義
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 位置エンコーディング行列の初期化
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model//2,)
        
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数インデックスにサイン波を適用
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数インデックスにコサイン波を適用
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model) -> (1, max_len, d_model)
        
        self.register_buffer('pe', pe) # 勾配を計算しない定数として登録
    
    def forward(self, x):
        # x: (シーケンス長, バッチサイズ, d_model)
        
        # 入力に位置エンコーディングを加算
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


#==========================================================#
#==========================================================#
#==========================================================#

# モデルのハイパーパラメータ設定
# これらの値はTransformerモデルの動作に影響を与える主要な要素です。

d_model = 512  # 埋め込みの次元数。各トークンをこのサイズのベクトルに変換します。
nhead = 8  # マルチヘッドアテンションのヘッド数。並列に異なるアテンションを計算します。
num_encoder_layers = 6  # エンコーダの層数。エンコーダに6つの層が積み重ねられます。
num_decoder_layers = 6  # デコーダの層数。デコーダにも6つの層が積み重ねられます。
dim_feedforward = 2048  # フィードフォワードネットワークの中間層の次元数。各トークンに対して適用されるMLPのサイズです。
dropout = 0.1  # ドロップアウト率。ネットワークの過学習を防ぐために、各層でランダムに一部のノードを無効化します。

# Transformerモデルのインスタンス化
# モデル全体を構築し、設定したハイパーパラメータを元にモデルを初期化します。

model = Transformer(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)

# 入力と出力のダミーデータ作成
# ここでは、モデルがどのように動作するかを確認するために、ランダムなダミーデータを生成しています。

src = torch.randint(0, 10000, (10, 32))  # 入力シーケンス (シーケンス長: 10, バッチサイズ: 32)
# torch.randint(0, 10000, (10, 32)) は、ボキャブラリー内のランダムなトークンIDを持つテンソルを生成します。
# 各トークンIDは0から9999の範囲にあります。

tgt = torch.randint(0, 10000, (10, 32))  # 出力シーケンス (シーケンス長: 10, バッチサイズ: 32)
# 出力シーケンスも同様にランダムに生成されています。

# モデルのフォワードパス
# 入力シーケンス (src) と出力シーケンス (tgt) をモデルに入力し、予測結果を得ます。

output = model(src, tgt)  # 出力テンソル (シーケンス長: 10, バッチサイズ: 32, ボキャブラリーサイズ: 10000)
# モデルは、入力シーケンス (src) と出力シーケンス (tgt) を受け取り、各ステップで予測される次のトークンの確率分布を返します。
# その結果は (シーケンス長, バッチサイズ, ボキャブラリーサイズ) のサイズのテンソルとして出力されます。

# 出力のサイズを確認
# モデルの出力が期待通りのサイズかどうかを確認します。

print(output.size())  # (10, 32, 10000)
# ここで出力されるのは、シーケンス長10、バッチサイズ32、ボキャブラリーサイズ10000に対応するテンソルのサイズです。
