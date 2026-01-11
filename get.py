#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用途：
    读取 query_labels.txt 的每一行，取该行的最后一个数/字段（支持逗号或空白分隔的最后一个 token），
    逐行写入到 query_filters.txt。

说明：
    - 不再从 shell 传参，输入/输出文件固定在 /home/wtj/data/arxiv 下。
    - 若某行为空或只有空白字符，则输出空行，保持行数一致。
"""

from __future__ import annotations

from pathlib import Path


DATA_DIR = Path("/home/wtj/data/arxiv")
INPUT_FILE = DATA_DIR / "query_labels.txt"
OUTPUT_FILE = DATA_DIR / "query_filters.txt"


def extract_last_token_per_line(input_path: Path, output_path: Path) -> None:
    # 逐行读取，提取每行最后一个 token 写入输出文件（兼容逗号/空白分隔）
    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                fout.write("\n")
                continue

            # 同时兼容逗号与空白分隔：先把逗号替换为空格，再按空白切分
            tokens = stripped.replace(",", " ").split()
            last_token = tokens[-1] if tokens else ""
            fout.write(f"{last_token}\n")


def main() -> int:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"数据目录不存在：{DATA_DIR}")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"输入文件不存在：{INPUT_FILE}")

    extract_last_token_per_line(INPUT_FILE, OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


