import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 데이터 load
real_data = np.load('/home/aix24702/jueunlee/seizure/data/blink_200_150_50_50_now/split_norm/real_private/pickles/data.p', allow_pickle=True)
real_labels = np.load('/home/aix24702/jueunlee/seizure/data/blink_200_150_50_50_now/split_norm/real_private/pickles/labels.p', allow_pickle=True)

# 데이터셋을 2차원 배열로 재구성
real_data_reshaped = real_data.reshape(real_data.shape[0], -1)

# t-SNE 모델을 초기화
tsne_model = TSNE(n_components=2, random_state=42)

# 실제 데이터에 대해 t-SNE를 실행
real_tsne = tsne_model.fit_transform(real_data_reshaped)

# 시각화를 위한 설정
plt.figure(figsize=(8, 8))

# 실제 데이터 (원 모양으로 표시)
for i in np.unique(real_labels):
    idx = real_labels == i
    marker = 'o'  # 원 모양
    color = 'blue' if i == 0 else 'red'  # 0은 파랑, 1은 빨강
    plt.scatter(real_tsne[idx, 0], real_tsne[idx, 1], marker=marker, color=color, label=f'Real Class {i}')

plt.title('t-SNE Visualization of Real Data')
plt.legend()

# 그림 저장
plt.savefig('real_data_tsne.png')
