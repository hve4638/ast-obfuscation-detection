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

# 검증 데이터 라벨링: 클래스 0과 1 각각 절반으로 라벨링
y_val = np.array([0 if i < len(validation_data) / 2 else 1 for i in range(len(validation_data))])

# 클래스 0, 클래스 1 각각 230개씩 샘플링
val_class_0_indices = np.where(y_val == 0)[0]
val_class_1_indices = np.where(y_val == 1)[0]

# 각 클래스의 개수를 출력하여 확인
print(f"Number of class 0 samples: {len(val_class_0_indices)}")
print(f"Number of class 1 samples: {len(val_class_1_indices)}")

# 샘플링 가능한지 확인하고 처리
if len(val_class_0_indices) < 230 or len(val_class_1_indices) < 230:
    raise ValueError("Not enough samples for class 0 or 1 to select 230 each.")

selected_class_0_indices = np.random.choice(val_class_0_indices, 230, replace=False)
selected_class_1_indices = np.random.choice(val_class_1_indices, 230, replace=False)
selected_indices = np.concatenate((selected_class_0_indices, selected_class_1_indices))

# 선택된 샘플을 기준으로 검증 데이터 선택
validation_data = [validation_data[i] for i in selected_indices]
y_val = y_val[selected_indices]

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
    for i, batch in enumerate(validation_loader):
        batch = batch.to(device)
        out = model(batch)
        pred = out.argmax(dim=1)
        y_true.extend(y_val[i*32:(i+1)*32])  # y_val에서 해당 배치에 해당하는 부분 추출
        y_pred.extend(pred.cpu().numpy())

# 테스트 데이터 평가 시간 측정 종료
eval_end_time = time.time()

# 결과 출력
print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))

# 전체 시간 측정 종료
total_end_time = time.time()

# 시간 출력
print(f"Total Time: {total_end_time - total_start_time:.4f} seconds")
print(f"Training Time: {train_end_time - train_start_time:.4f} seconds")
print(f"Evaluation Time: {eval_end_time - eval_start_time:.4f} seconds")
