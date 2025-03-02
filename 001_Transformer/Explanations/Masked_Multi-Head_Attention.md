**Masked Multi-Head Attention** は、Transformerモデルの中でも特にデコーダ部分で使用されるメカニズムで、主に自己回帰的なタスク、例えば文章生成や翻訳などにおいて重要な役割を果たします。このセクションでは、Masked Multi-Head Attentionの原理とその意図について詳しく説明します。

### 1. Multi-Head Attention の基本原理

まず、**Multi-Head Attention** の基本原理を復習します。

- **Multi-Head Attention** では、クエリ (Q)、キー (K)、バリュー (V) の3つの入力が用いられます。
- 各クエリと全てのキーとの類似度を計算し、ソフトマックスで正規化した重みをバリューに適用することで、出力が生成されます。
- これを複数のヘッド（異なる重み行列）で並列に行い、最終的にこれらの出力を結合して一つの出力を得ます。

### 2. Masked Multi-Head Attention の必要性

**Masked Multi-Head Attention** は、主にデコーダ部分で使用され、以下の理由で必要となります：

#### a. 自己回帰モデルにおける未来の情報の遮断

- デコーダは、シーケンスを左から右に生成する際、**現在のトークンを生成する際に、未来のトークンの情報を参照してはならない**という制約があります。
- 例えば、ある文章を生成する際に、まだ生成していない未来の単語を参照してしまうと、モデルがその単語を利用してしまい、正しい生成ができなくなります。

#### b. 未来のトークンのマスク

- これを防ぐために、**マスク** を用いて、未来のトークンに対する注意を遮断します。具体的には、各クエリ（現在のトークン）が、過去および現在のトークンにのみ注意を払うようにし、未来のトークンには注意を向けられないようにします。

### 3. Masked Multi-Head Attention の実装

Masked Multi-Head Attention は、以下のように実装されます：

1. **注意スコアの計算**:
   - クエリとキーの内積を計算して、注意スコアを得ます。これにより、各トークンが他のどのトークンに注意を向けるべきかが決まります。

2. **マスクの適用**:
   - 次に、未来のトークンに対する注意スコアを無効化するために、**マスク行列**を適用します。このマスク行列では、未来のトークンに対するスコアに大きな負の値（例えば、\(-\infty\)）を加えます。これにより、ソフトマックス関数を適用した際に、これらのスコアがゼロに近い値になるようにします。

3. **ソフトマックスの適用と重み付け**:
   - マスクされた注意スコアにソフトマックス関数を適用して正規化し、バリューに対して重み付けを行います。

4. **複数のヘッドで並列計算**:
   - このプロセスを複数のヘッドで並列に行い、最終的に各ヘッドの出力を結合して一つの出力を得ます。

### 4. Masked Multi-Head Attention の意図

Masked Multi-Head Attention の意図は、自己回帰的なタスクにおいてモデルが未来の情報を参照するのを防ぐことです。具体的には、次のような意図があります：

- **モデルの正しい学習**:
  - デコーダがまだ生成されていない未来のトークンを参照することを防ぐことで、モデルが現在のトークンのみを基に次のトークンを正しく予測できるようにします。

- **現実的な生成シナリオの再現**:
  - 文章生成や翻訳のタスクでは、現実的には未来のトークンがわからない状態で次のトークンを予測する必要があるため、Masked Multi-Head Attention によってこの制約が再現されます。

- **過去の情報の利用**:
  - 過去および現在のトークンにのみ注意を向けることで、過去の文脈に基づいて次のトークンを予測します。

### まとめ

**Masked Multi-Head Attention** は、Transformerのデコーダにおいて、自己回帰的なタスクで未来の情報を参照しないようにするための重要なメカニズムです。これにより、モデルは過去および現在の情報に基づいて次のトークンを正確に予測でき、未来のトークンを不正に参照することを防ぎます。このメカニズムは、文章生成や翻訳などのタスクで正確で現実的な予測を行うために不可欠です。
