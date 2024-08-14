#ハイブリッドアプローチの例
#たとえば、交差点での右折時に対向車がいる場合に優先権を与えるルールを組み込みつつ、
#データ駆動型モデルでその場の状況に応じた微調整を行うといったアプローチです。

# ハイブリッドアプローチの例
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)  # 出力は回避アクションとアクセラレーション

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        outputs = torch.sigmoid(self.fc2(x))
        
        # ルールベースの処理: 右折時に対向車がいる場合、停止する
        if self.is_right_turn(x) and self.detect_oncoming_vehicle(x):
            outputs[:, 0] = 0  # 回避アクションを強制的に停止にセット
        
        return outputs

    def is_right_turn(self, x):
        # 右折の条件（例）
        return x[:, 0] > 0.5

    def detect_oncoming_vehicle(self, x):
        # 対向車の検出（例）
        return x[:, 1] > 0.5
