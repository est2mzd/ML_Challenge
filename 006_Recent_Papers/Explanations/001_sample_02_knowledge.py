import torch
import torch.nn as nn
import torch.optim as optim

class KnowledgeDrivenPedestrianModel(nn.Module):
    def __init__(self):
        super(KnowledgeDrivenPedestrianModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)  # 出力は歩行者がいるかどうかの2クラス

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        pedestrian_prob = torch.sigmoid(self.fc2(x))

        # 知識ベースのルールを適用：横断歩道に歩行者がいる場合は必ず停止
        stop_decision = (pedestrian_prob[:, 0] > 0.5).float()  # 0.5以上の確率で歩行者がいる場合は停止
        return stop_decision, pedestrian_prob

# 知識駆動型データセット（データ駆動型と同じ）
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 10)
        self.labels = torch.randint(0, 2, (1000, 2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# モデル、損失関数、最適化器の定義（データ駆動型と同じ）
model = KnowledgeDrivenPedestrianModel()
criterion = nn.BCELoss()  # バイナリ分類タスク
optimizer = optim.Adam(model.parameters(), lr=0.001)

# データローダーの準備（データ駆動型と同じ）
dataset = SimpleDataset()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# トレーニングループ
for epoch in range(10):  # 10エポック
    for inputs, labels in train_loader:
        stop_decision, outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

# このコードでは、歩行者がいるかどうかの認識に加えて、歩行者がいる場合に車両を停止するという知識が直接組み込まれています。
