import os
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.data_utils import sixd_to_rotmat_torch, convert_rot_mats_to_rot_vecs

if __name__ == "__main__":
    actions = ['posing', 'greeting', 'sitting', 'walking', 'smoking', 'walkingtogether',
               'phoning', 'walkingdog', 'waiting', 'eating', 'discussion', 'purchases',
               'sittingdown', 'directions', 'takingphoto']
    data_dir = '../data/h3.6m'
    path_to_data = os.path.join(data_dir, 'h36m_train_75.npz')
    data = np.load(path_to_data, allow_pickle=True)

    all_sixd_data = data['sampled_sixd_seq']
    label_seqs = data['label_seq']
    N = len(all_sixd_data)  # 样本数
    V = all_sixd_data[0].shape[1]  # 关节数

    # 创建输出目录
    output_dir = './frame_cluster_visualization'
    os.makedirs(output_dir, exist_ok=True)

    for j in tqdm(range(V), desc="Processing joints"):
        # 1. 收集所有样本的当前关节的每一帧作为独立样本
        all_frames = []
        sample_indices = []  # 记录帧来源的样本索引
        for sample_idx in range(N):
            sixd_seq = all_sixd_data[sample_idx][:, j, :]  # (T, 6)
            all_frames.append(sixd_seq)
            sample_indices.extend([sample_idx] * sixd_seq.shape[0])

        if not all_frames:
            continue  # 无有效数据
        all_frames = np.concatenate(all_frames, axis=0)  # (Total_Frames, 6)
        sample_indices = np.array(sample_indices)

        # 2. 标准化数据（6维特征）
        scaler = StandardScaler()
        scaled_frames = scaler.fit_transform(all_frames)

        # 3. 聚类（调整min_cluster_size以适应单帧数据）
        clusterer = HDBSCAN(min_cluster_size=20)  # 减小最小簇大小
        cluster_labels = clusterer.fit_predict(scaled_frames)

        # 4. 提取聚类中心（排除噪声点-1）
        unique_labels = np.unique(cluster_labels[cluster_labels != -1])
        centers = []
        for label in unique_labels:
            mask = (cluster_labels == label)
            cluster_data = scaled_frames[mask]
            centers.append(np.mean(cluster_data, axis=0))
        if not centers:
            continue  # 无有效簇
        centers = np.array(centers)

        # 5. PCA降维至3D（从6维降到3维）
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_frames)
        centers_pca = pca.transform(scaler.transform(centers))

        # 6. 可视化聚类结果
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
            c=cluster_labels, cmap='viridis', s=5, alpha=0.6,
            label='Frames'
        )
        ax.scatter(
            centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2],
            c='red', marker='X', s=200, label='Cluster Centers'
        )

        # 7. 随机选取10个样本的帧并高亮显示
        selected_samples = np.random.choice(N, 10, replace=False)
        selected_mask = np.isin(sample_indices, selected_samples)
        ax.scatter(
            pca_result[selected_mask, 0], pca_result[selected_mask, 1], pca_result[selected_mask, 2],
            c='black', s=30, edgecolors='white', linewidth=0.5,
            label='Selected Samples'
        )

        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        plt.title(f'Joint {j} Clustering (Total Clusters: {len(unique_labels)})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'joint_{j}_frame_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()

print("单帧聚类与可视化完成！结果保存在", output_dir)