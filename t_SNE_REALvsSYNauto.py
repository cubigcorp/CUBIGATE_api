import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 실제 데이터 로드 (이 부분은 변경하지 않음)
real_data = np.load('/home/aix24702/jueunlee/seizure/data/blink_150_100_100_100/split_norm/real_private/pickles/data.p', allow_pickle=True)
real_labels = np.load('/home/aix24702/jueunlee/seizure/data/blink_150_100_100_100/split_norm/real_private/pickles/labels.p', allow_pickle=True)
real_data_reshaped = real_data.reshape(200, -1)

# 합성 데이터 경로의 기본 형태
synthetic_data_base_path = '/home/aix24702/jueunlee/seizure/data/blink_150_100_100_100/syn/result_2_1_linear/pickles/{}/data.p'
synthetic_labels_base_path = '/home/aix24702/jueunlee/seizure/data/blink_150_100_100_100/syn/result_2_1_linear/pickles/{}/labels.p'

# 0부터 17까지의 데이터 세트에 대해 반복
for i in range(18):
    # 합성 데이터 로드
    synthetic_data_path = synthetic_data_base_path.format(i)
    synthetic_labels_path = synthetic_labels_base_path.format(i)
    synthetic_data = np.load(synthetic_data_path, allow_pickle=True)
    synthetic_labels = np.load(synthetic_labels_path, allow_pickle=True)

    # 데이터셋을 2차원 배열로 재구성
    synthetic_data_reshaped = synthetic_data.reshape(200, -1)

    # t-SNE 모델 초기화 및 데이터 합치기
    tsne_model = TSNE(n_components=2, random_state=42)
    combined_data = np.vstack((synthetic_data_reshaped, real_data_reshaped))

    # 합친 데이터에 대해 t-SNE를 실행
    combined_tsne = tsne_model.fit_transform(combined_data)

    # 시각화를 위한 설정
    num_synthetic_samples = synthetic_data_reshaped.shape[0]
    plt.figure(figsize=(8, 8))

    # 합성 데이터 (네모 모양으로 표시)
    for j in np.unique(synthetic_labels):
        idx = synthetic_labels == j
        plt.scatter(combined_tsne[:num_synthetic_samples][idx, 0], combined_tsne[:num_synthetic_samples][idx, 1], marker='s', color='blue' if j == 0 else 'red', label=f'Synthetic Class {j}')

    # 실제 데이터 (원 모양으로 표시)
    offset = num_synthetic_samples
    for j in np.unique(real_labels):
        idx = real_labels == j
        plt.scatter(combined_tsne[num_synthetic_samples:][idx, 0], combined_tsne[num_synthetic_samples:][idx, 1], marker='o', color='blue' if j == 0 else 'red', label=f'Real Class {j}')

    plt.title(f't-SNE Visualization for Set {i}')
    plt.legend()

    # 그림 저장
    plt.savefig(f'/home/aix24702/jueunlee/seizure/data/blink_200_150_50_50_now/PNG/no_demo/relative/sne{i}.png')
    plt.close()  # 현재 그림 닫기
