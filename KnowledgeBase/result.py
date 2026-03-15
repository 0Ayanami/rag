import json
from collections import defaultdict
from statistics import mean
"统计json文件中每个task的有效样本数和各项指标的平均值（只统计raw_answer非空的样本）"

with open("./qwen.json", encoding="utf-8") as f:
    data = json.load(f)

# 要统计的字段
metrics = [
    "事实准确性_大模型",
    "事实准确性_人工",
    "回答相关性_大模型",
    "回答相关性_人工",
    "检索相关性"
]

# 按 task 分组统计（只统计 raw_answer 非空）
task_stats = defaultdict(lambda: {m: [] for m in metrics})

for item in data:
    if not item.get("raw_answer", "").strip():  # 跳过空回答
        continue
    task = item["task"]
    for metric in metrics:
        value = item.get(metric, 0)
        task_stats[task][metric].append(value)

# 计算平均值并打印表格
print("任务     | 有效样本 | " + " | ".join(f"{m:>12}" for m in metrics))
print("-" * 100)

for task, stats in sorted(task_stats.items()):
    counts = len(stats[metrics[0]])
    if counts == 0:
        continue
    avgs = {m: round(mean(stats[m]), 2) for m in metrics}
    print(f"{task:<8} | {counts:>6}   | " + " | ".join(f"{avgs[m]:>12.2f}" for m in metrics))