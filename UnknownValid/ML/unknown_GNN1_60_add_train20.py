import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import classification_report

# TensorFlow 및 GPU 확인
import tensorflow as tf
print("TensorFlow CUDA Available:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# 정상 난독화 및 악성 난독화 데이터 경로 설정
obfuscated_normal_dir_1 = "D:/PowerShell/1119/extractedmatrix/extracted(60)/IseSteroids/npy"
obfuscated_normal_dir_2 = "D:/PowerShell/1119/extractedmatrix/extracted(60)/invokeObfuscation/npy"
obfuscated_normal_dir_3 = "D:/PowerShell/1119/extractedmatrix/extracted(60)/IseSteroids/npy"
attack_dir_2 = "D:/PowerShell/1119/extractedmatrix/extracted(60)/EncodedPowershell(obfuscated)/npy"

# 검증 데이터 경로 설정
validation_dir = "D:/PowerShell/unknownvalidationset/extracted(60)/rc_powershell/npy"

# .npy 파일에서 인접 행렬 데이터를 불러오는 함수
def load_data_from_npy(directory, label):
    data_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            adj_matrix = np.load(file_path)

            # 인접 행렬에서 엣지 인덱스 추출
            edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
            features = torch.eye(adj_matrix.shape[0], dtype=torch.float)  # Identity matrix as node features

            data = Data(x=features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
            data_list.append(data)
    return data_list

# 데이터 불러오기: 정상 난독화와 악성 난독화
normal_data = load_data_from_npy(obfuscated_normal_dir_1, label=0) + \
              load_data_from_npy(obfuscated_normal_dir_2, label=0) + \
              load_data_from_npy(obfuscated_normal_dir_3, label=0)
malicious_data = load_data_from_npy(attack_dir_2, label=1)
all_data = normal_data + malicious_data

# 검증 데이터 불러오기: Unknown 검증 데이터셋
validation_data = load_data_from_npy(validation_dir, label=-1)  # 라벨은 학습에 사용되지 않음

# GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 그래프 수준의 집계를 수행
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 전체 시간 측정 시작
total_start_time = time.time()

# 모델 학습 및 평가 루프
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(all_data, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)

model = GCN(input_dim=all_data[0].num_features, hidden_dim=64, output_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습 시간 측정 시작
train_start_time = time.time()

# 학습 루프
model.train()
for epoch in range(10):  # 10 에포크 동안 학습
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()

# 학습 시간 측정 종료
train_end_time = time.time()

# 테스트 데이터 평가 시간 측정 시작
eval_start_time = time.time()

# 테스트 데이터 평가 (Unknown 검증 데이터셋 사용)
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in validation_loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.argmax(dim=1)
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

# 테스트 데이터 평가 시간 측정 종료
eval_end_time = time.time()

# 결과 출력
y_true = [0 if label == -1 else label for label in y_true]  # 임시 라벨을 0으로 변경하여 평가
y_pred = [0 if label == -1 else label for label in y_pred]
print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))

# 그래프 시각화 (정상 난독화와 악성 난독화 각각 하나씩 시각화, 방향이 있게 설정)
data_to_visualize_normal = normal_data[0]
G_normal = to_networkx(data_to_visualize_normal, to_undirected=False)  # 방향 있는 그래프
plt.figure(figsize=(8, 6))
nx.draw(G_normal, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', arrows=True)
plt.title("Graph Visualization of a Normal Obfuscated Sample (Directed)")
plt.show()

data_to_visualize_malicious = malicious_data[0]
G_malicious = to_networkx(data_to_visualize_malicious, to_undirected=False)  # 방향 있는 그래프
plt.figure(figsize=(8, 6))
nx.draw(G_malicious, with_labels=True, node_color='salmon', node_size=500, edge_color='gray', arrows=True)
plt.title("Graph Visualization of a Malicious Obfuscated Sample (Directed)")
plt.show()

# 검증 데이터셋에서 샘플 하나 시각화
data_to_visualize_validation = validation_data[0]
G_validation = to_networkx(data_to_visualize_validation, to_undirected=False)  # 방향 있는 그래프
plt.figure(figsize=(8, 6))
nx.draw(G_validation, with_labels=True, node_color='lightgreen', node_size=500, edge_color='gray', arrows=True)
plt.title("Graph Visualization of a Validation Sample (Directed)")
plt.show()

# 전체 시간 측정 종료
total_end_time = time.time()

# 시간 출력
print(f"Total Time: {total_end_time - total_start_time:.4f} seconds")
print(f"Training Time: {train_end_time - train_start_time:.4f} seconds")
print(f"Evaluation Time: {eval_end_time - eval_start_time:.4f} seconds")
