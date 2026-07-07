import re
import matplotlib.pyplot as plt

logs = {
    # "beta=0.0":   "/home/wtj/Beta-DiskANN/build/demo/out/search_k.txt",
    "global":  "/home/wangtj/results/tripclick/global_selectivity.txt",
    # "beta=0.8": "/home/wtj/Beta-DiskANN/build/demo/out/search_beta0.8_k.txt",
    "cor": "/home/wangtj/results/tripclick/COR1.txt",
}

def parse_log(path):
    qps_list, r10_list = [], []
    pattern = re.compile(r'^\s*\d+')  # 行首数字(Ls)的行
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not pattern.match(line):
                continue
            tokens = line.split()
            # 期望: [Ls, QPS, AvgCmps, MeanLatency, P99, R1, ..., R10]
            if len(tokens) < 5 + 1:
                continue
            try:
                qps = float(tokens[1])
                r10 = float(tokens[-1])  # 最后一列是 Recall@10
            except ValueError:
                continue
            qps_list.append(qps)
            r10_list.append(r10)
    return qps_list, r10_list

plt.figure(figsize=(8,6))
for name, path in logs.items():
    qps, r10 = parse_log(path)
    # 以 Recall@10 为横轴，按横轴排序便于连线
    pairs = sorted(zip(r10, qps), key=lambda x: x[0])
    if not pairs:
        continue
    r10_sorted, qps_sorted = zip(*pairs)
    plt.plot(r10_sorted, qps_sorted, '-o', label=name)

plt.xlabel("Recall@10")
plt.ylabel("QPS")
plt.title("QPS vs Recall@10")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
out_png = "/home/wangtj/results/tripclick/compare1.png"
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print("Saved:", out_png)