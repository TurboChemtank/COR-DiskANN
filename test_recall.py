import sys
import numpy as np
sys.path.append('/home/wtj/DiskANN/python/apps')
from utils import bin_to_numpy, read_gt_file, calculate_recall, calculate_recall_from_gt_file

# 方法1: 如果你有查询结果文件（.bin格式）
def test_recall_from_bin(result_file, gt_file, K=10):
    """从二进制结果文件计算召回率"""
    result_ids, _ = read_gt_file(result_file)
    recall = calculate_recall_from_gt_file(K, result_ids, gt_file)
    print(f"Recall@{K}: {recall:.4f}")
    return recall

# 方法2: 如果你有numpy数组格式的结果
def test_recall_from_arrays(result_indices, truth_indices, K=10):
    """从numpy数组计算召回率"""
    recall = calculate_recall(result_indices, truth_indices, recall_at=K)
    print(f"Recall@{K}: {recall:.4f}")
    return recall

# 方法3: 读取文本格式的ground truth
def load_txt_groundtruth(filename):
    """加载文本格式的ground truth（逗号分隔）"""
    gt = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                neighbors = [int(x) for x in line.strip().split(',')]
                gt.append(neighbors)
    return np.array(gt)

# 示例使用
if __name__ == '__main__':
    # 测试不同的K值
    k_values = [1, 5, 10, 20, 50, 100]
    
    # 加载ground truth
    gt_file = '/home/wtj/Test-Lable/arxiv_label_base.txt'
    gt = load_txt_groundtruth(gt_file)
    
    print(f"Ground truth shape: {gt.shape}")
    print(f"测试查询数: {gt.shape[0]}")
    
    # 这里需要你的查询结果
    # results = your_search_results  # shape: (num_queries, K)
    
    # for K in k_values:
    #     recall = calculate_recall(results, gt, recall_at=K)
    #     print(f"Recall@{K}: {recall:.4f}")
