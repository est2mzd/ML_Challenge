'''
コードの説明
入力シーケンス (Input Sequence):

3つの商品があり、それぞれの特徴ベクトルは [色 (Red, Green, Blue), 重さ (Weight)] で表されます。例として、[1.0, 0.0, 0.0, 0.5] は赤色で重さが0.5kgの商品の特徴を示しています。
クエリ (Query), キー (Key), バリュー (Value):

Self Attentionでは、クエリ、キー、バリューはすべて同じ入力から生成されます。つまり、すべてのトークンが他のすべてのトークンに注意を向け、その相関に基づいて自分の特徴を再計算します。
クエリとキーの類似度を計算:

クエリとキーの間で内積を取り、類似度を計算します。この内積の結果として、各トークンが他のトークンとどれだけ似ているかを示すスコアが得られます。例えば、赤い色で重さが似ている商品同士は高いスコアを持つでしょう。
スケーリング:

類似度スコアをキーの次元数の平方根で割ってスケーリングします。これにより、スコアが過度に大きくならないように調整されます。
ソフトマックス関数で正規化:

スケーリングされたスコアをソフトマックス関数で正規化し、各トークンが他のトークンに対してどれだけの注意を向けるべきかを決定します。これにより、注意の重みが計算されます。
バリューに注意の重みを適用:

計算された注意の重みを使って、各トークンが他のトークンから受け取る情報を調整します。具体的には、各トークンのバリューに重みを掛け合わせ、最終的な出力を得ます。
実行結果の解釈
Similarity Scores:

各商品の特徴ベクトル間の類似度を示します。同じ商品や類似した特徴を持つ商品は、高いスコアを持ちます。
Scaled Scores:

類似度スコアをスケーリングしたものです。
Attention Weights:

各商品の特徴ベクトルに対する注意の重みです。特定のトークンが他のトークンにどれだけ注意を向けるべきかを示しています。
Weighted Values (Output):

最終的な出力です。各トークンの特徴は、他のトークンとの相関に基づいて更新され、より情報豊かな特徴ベクトルに変換されます。
まとめ
このSelf Attentionの例では、各商品（トークン）が他のすべての商品に対してどれだけ関連があるかを計算し、その関連性に基づいて自身の特徴を再計算しています。これは、入力シーケンス内の情報をより豊かに捉えるために使用される強力なメカニズムです。
'''

import torch
import torch.nn.functional as F

# 入力シーケンスの定義
# ここでは、物理的に意味のある値として、異なる商品の特徴ベクトルを考えます。
# 例えば、3つの商品があるとします。

# 商品の特徴量: 色 (Red, Green, Blue) と重さ (Weight)
input_sequence = torch.tensor([
    [1.0, 0.0, 0.0, 0.5],  # 商品1: 赤い色、重さ0.5kg (Red=1, Green=0, Blue=0, Weight=0.5)
    [0.0, 1.0, 0.0, 0.6],  # 商品2: 緑色、重さ0.6kg (Red=0, Green=1, Blue=0, Weight=0.6)
    [1.0, 0.0, 0.0, 0.7]   # 商品3: 赤い色、重さ0.7kg (Red=1, Green=0, Blue=0, Weight=0.7)
])  # サイズ: (3, 4)

# クエリ、キー、バリューの生成
# Self Attentionでは、クエリ、キー、バリューはすべて同じ入力から生成されます。
queries = input_sequence  # クエリは入力シーケンスそのもの
keys    = input_sequence     # キーも入力シーケンスそのもの
values  = input_sequence   # バリューも入力シーケンスそのもの

# クエリとキーの類似度を計算 (内積を取る)
similarity_scores = torch.matmul(queries, keys.T)  # サイズ: (3, 3)
print(f"Similarity Scores:\n{similarity_scores}")

# スケーリング (キーの次元数の平方根で割る)
d_k = keys.size(1)  # キーの次元数 (この場合は4)
scaled_scores = similarity_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
print(f"Scaled Scores:\n{scaled_scores}")

# ソフトマックス関数で正規化して注意の重みを得る
attention_weights = F.softmax(scaled_scores, dim=-1)  # サイズ: (3, 3)
print(f"Attention Weights:\n{attention_weights}")

# 注意の重みをバリューに適用する
weighted_values = torch.matmul(attention_weights, values)  # サイズ: (3, 4)
print(f"Weighted Values (Output):\n{weighted_values}")
