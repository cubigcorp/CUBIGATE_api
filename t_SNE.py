import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 데이터 load
synthetic_data = np.load('/home/aix24702/jueunlee/seizure/data/chbmit_synthetic/pickles/17/data.p', allow_pickle=True)
real_data = np.load('/home/aix24702/jueunlee/seizure/data/chbmit_synthetic/real_eval/real_pickle/data.p', allow_pickle=True)

# 각 데이터에 대한 클래스 구분자(라벨) load
synthetic_labels = np.load('/home/aix24702/jueunlee/seizure/data/chbmit_synthetic/pickles/17/labels.p', allow_pickle=True)
real_labels = np.load('/home/aix24702/jueunlee/seizure/data/chbmit_synthetic/real_eval/real_pickle/labels.p', allow_pickle=True)

# 데이터셋을 2차원 배열로 재구성
synthetic_data_reshaped = synthetic_data.reshape(100, -1)
real_data_reshaped = real_data.reshape(100, -1)

# t-SNE 모델을 초기화하고 두 데이터셋을 합침 (그냥 한 표에 그려넣기 위함)
tsne_model = TSNE(n_components=2, random_state=0)
combined_data = np.vstack((synthetic_data_reshaped, real_data_reshaped))

# 합친 데이터에 대해 t-SNE를 실행
combined_tsne = tsne_model.fit_transform(combined_data)

# t-SNE 결과를 라벨에 따라 시각화
num_synthetic_samples = synthetic_data_reshaped.shape[0]
plt.figure(figsize=(8, 8))

# 합성 데이터를 라벨에 따라 색상을 다르게 하여 플롯
for i in np.unique(synthetic_labels):
    idx = synthetic_labels == i
    plt.scatter(combined_tsne[:num_synthetic_samples][idx, 0], combined_tsne[:num_synthetic_samples][idx, 1], label=f'Synthetic Class {i}')

# 실제 데이터를 라벨에 따라 색상을 다르게 하여 플롯
offset = num_synthetic_samples
for i in np.unique(real_labels):
    idx = real_labels == i
    plt.scatter(combined_tsne[num_synthetic_samples:][idx, 0], combined_tsne[num_synthetic_samples:][idx, 1], label=f'Real Class {i}')

plt.title('t-SNE Visualization')
plt.legend()

# 그림을 sne.png로 저장
plt.savefig('sne.png')
