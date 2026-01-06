import pickle
import argparse
from pathlib import Path
import numpy as np

# -------------------------
# 参数
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='llama3')
parser.add_argument("--dataset", type=str, default='ml1m')
args = parser.parse_args()

MODEL = args.model
DATASET = args.dataset
BASE_DIR = Path(f"./repeat_defense/{DATASET}/{MODEL}")

k_values = [1]
thresholds = np.linspace(1, 10, 10)

def analyze(split: str):
    print(f"\n===== {split.upper()} =====")

    for k in k_values:
        with open(BASE_DIR / split / f"{k}_shots" / "member.pkl", "rb") as f:
            member_data = np.array(pickle.load(f))
            print(len(member_data))

        with open(BASE_DIR / split / f"{k}_shots" / "nonmember.pkl", "rb") as f:
            nonmember_data = np.array(pickle.load(f))

        best_threshold = None
        best_acc = -1.0
        best_stats = None

        for threshold in thresholds:
            TP = np.sum(member_data >= threshold)
            FN = np.sum(member_data < threshold)

            TN = np.sum(nonmember_data < threshold)
            FP = np.sum(nonmember_data >= threshold)

            # rates
            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            acc = (TP + TN) / (TP + TN + FP + FN)

            # precision / recall / F1
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = tpr
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # 👉 目标：最大 Accuracy
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
                best_stats = (tpr, tnr, acc, f1)

        tpr, tnr, acc, f1 = best_stats
        advantage = 2 * (acc - 0.5)

        print(
            f"{k}_shots | "
            f"best_th={best_threshold:.4f} | "
            f"TPR={tpr:.4f} | "
            f"TNR={tnr:.4f} | "
            f"ACC={acc:.4f} | "
            f"F1={f1:.4f} | "
            f"ADV={advantage:.4f}"
        )

# -------------------------
# start / end
# -------------------------
analyze("start")

