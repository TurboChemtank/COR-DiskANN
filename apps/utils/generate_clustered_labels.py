#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import struct
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import ctypes
import time

# ==========================================
# 全局变量 (仅在子进程初始化时绑定，避免在进程间拷贝海量数据)
# ==========================================
global_vectors_mmap = None
global_labels_per_point = None
global_mean_sq_dist = None
global_npts = None
global_dim = None


def read_fbin_header(filename):
    """只读取文件头，不加载数据"""
    with open(filename, "rb") as f:
        npts = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
    return npts, dim


def init_worker(fbin_path, npts, dim, shared_penalty_array, estimated_mean_dist):
    """
    子进程初始化函数：
    每个子进程都会执行一次，利用 memmap 建立对 50GB 文件的 Zero-copy 映射。
    """
    global global_vectors_mmap
    global global_labels_per_point
    global global_mean_sq_dist
    global global_npts
    global global_dim

    global_npts = npts
    global_dim = dim
    global_mean_sq_dist = estimated_mean_dist

    # 建立只读的内存映射
    offset = 8  # 跳过 fbin 头的 8 字节 (npts, dim)
    global_vectors_mmap = np.memmap(fbin_path, dtype=np.float32, mode="r", offset=offset, shape=(npts, dim))

    # 绑定无锁共享内存，将其转换为可以直接操作的 numpy 数组
    global_labels_per_point = np.ctypeslib.as_array(shared_penalty_array)


def process_single_label(task_info):
    """
    核心工作函数：为一个特定的标签在 1 亿个点中寻找归宿
    """
    label_id, count, gamma, penalty_weight = task_info

    if count <= 0:
        return (label_id, [])

    # 安全保护：无放回采样时 count 不能超过点数
    count = min(int(count), int(global_npts))

    # 1. 采样锚点 (Anchor)
    anchor_idx = int(np.random.randint(0, global_npts))
    anchor_vec = global_vectors_mmap[anchor_idx]

    # 2. 分块计算 (Chunking) 以防止单次计算挤爆内存
    chunk_size = 5_000_000  # 每次计算 500 万点，峰值内存约 20MB
    penalized_weights = np.empty(global_npts, dtype=np.float64)

    for start in range(0, global_npts, chunk_size):
        end = min(start + chunk_size, global_npts)

        vec_chunk = global_vectors_mmap[start:end]

        # 计算距离平方
        sq_dists = np.sum((vec_chunk - anchor_vec) ** 2, axis=1)

        # 计算基础高斯权重
        weights = np.exp(-gamma * (sq_dists / global_mean_sq_dist))

        # 施加共享的容量惩罚
        local_penalty = global_labels_per_point[start:end]
        penalized_weights[start:end] = weights / (1.0 + penalty_weight * local_penalty)

    # 极小扰动防止除零异常
    penalized_weights += 1e-12

    # 3. 概率归一化
    prob_dist = penalized_weights / np.sum(penalized_weights)

    # 4. 加权随机采样
    chosen_indices = np.random.choice(global_npts, size=count, replace=False, p=prob_dist)

    # 5. 更新共享惩罚数组 (无锁直接写入，极小的写冲突完全可接受)
    for idx in chosen_indices:
        global_labels_per_point[idx] += 1.0

    return (f"L_{label_id}", chosen_indices.tolist())


def estimate_mean_sq_dist(fbin_path, npts, dim, sample_size=100_000):
    """预估高斯核的距离尺度因子，避免子进程重复计算"""
    print("Estimating global scale factor for Gaussian RBF...")
    offset = 8
    mmap_vecs = np.memmap(fbin_path, dtype=np.float32, mode="r", offset=offset, shape=(npts, dim))

    sample_indices = np.random.choice(npts, size=min(sample_size, npts), replace=False)
    sample_vecs = mmap_vecs[sample_indices]

    anchor = sample_vecs[0]
    dists = np.sum((sample_vecs - anchor) ** 2, axis=1)
    mean_dist = float(np.mean(dists))
    return mean_dist if mean_dist > 0 else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .fbin vector file")
    parser.add_argument("--out", type=str, default="synthetic_labels_100M.txt")
    parser.add_argument("--num_labels", type=int, default=100_000)
    parser.add_argument("--zipf", type=float, default=1.5)
    parser.add_argument("--avg_labels", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU threads to use")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. 提取元数据
    npts, dim = read_fbin_header(args.data)
    total_assignments = npts * args.avg_labels
    print(f"Dataset: {npts} points, {dim} dimensions.")
    print(f"Starting parallel engine with {args.workers} workers...")

    # 2. 尺度预估
    estimated_mean_dist = estimate_mean_sq_dist(args.data, npts, dim)

    # 3. Zipf 目标计算
    print("Generating Zipfian target counts...")
    ranks = np.arange(1, args.num_labels + 1)
    probabilities = 1.0 / (ranks ** args.zipf)
    probabilities /= np.sum(probabilities)
    target_counts = np.round(probabilities * total_assignments).astype(int)
    target_counts = np.maximum(target_counts, 1)

    # 4. 构建任务列表 (随机打乱标签顺序，防止头部超级大标签全挤在一个进程里)
    tasks = []
    label_order = np.random.permutation(args.num_labels)
    for l in label_order:
        tasks.append((l, int(target_counts[l]), args.gamma, args.penalty))

    # 5. 分配底层 C 语言共享内存 (400MB 左右)
    print("Allocating lock-free shared memory for capacity penalty...")
    shared_penalty = RawArray(ctypes.c_float, npts)

    # 6. 多进程狂飙
    print("Running Thomas Cluster Process in parallel...")
    start_time = time.time()

    point_labels = [[] for _ in range(npts)]

    # 使用 spawn 模式在各种 OS 下都比较稳健，避免 fork 复制父进程的内存
    ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
    with ctx.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.data, npts, dim, shared_penalty, estimated_mean_dist),
    ) as pool:
        # imap_unordered 不保证返回顺序，速度最快
        for label_name, chosen_indices in tqdm(pool.imap_unordered(process_single_label, tasks), total=len(tasks)):
            for idx in chosen_indices:
                point_labels[idx].append(label_name)

    print(f"Parallel Generation completed in {(time.time() - start_time) / 60:.2f} minutes.")

    # 7. 孤儿点兜底补全
    print("Executing orphan node fallback...")
    orphan_count = 0
    for idx in range(npts):
        if len(point_labels[idx]) == 0:
            random_label_idx = np.random.choice(args.num_labels, p=probabilities)
            point_labels[idx].append(f"L_{random_label_idx}")
            orphan_count += 1
    print(f"Fixed {orphan_count} orphan points.")

    # 8. 极速批量落盘
    print(f"Writing dataset to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as f:
        batch_size = 200_000
        lines_buffer = []
        for labels in tqdm(point_labels, desc="Disk I/O"):
            lines_buffer.append(",".join(labels) + "\n")
            if len(lines_buffer) >= batch_size:
                f.writelines(lines_buffer)
                lines_buffer.clear()
        if lines_buffer:
            f.writelines(lines_buffer)

    print("All tasks finished perfectly!")


if __name__ == "__main__":
    main()
