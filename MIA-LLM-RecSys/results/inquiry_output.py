import pickle
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='llama3', required=True)
parser.add_argument("--dataset", type=str, default='ml1m', required=True)
args = parser.parse_args()

MODEL = args.model
DATASET = args.dataset
BASE_DIR = Path(f"./inquiry/{DATASET}/{MODEL}")
k_values = [1, 5, 10]

def analyze(split: str):
    for k in k_values:
        with open(BASE_DIR / split / f"{k}_shots" / "member.pkl", "rb") as f:
            member_data = pickle.load(f)

        with open(BASE_DIR / split / f"{k}_shots" / "nonmember.pkl", "rb") as f:
            nonmember_data = pickle.load(f)

        # confusion matrix
        TP = sum(1 for x in member_data if x == 1)
        FN = sum(1 for x in member_data if x == 0)

        TN = sum(1 for x in nonmember_data if x == 0)
        FP = sum(1 for x in nonmember_data if x == 1)

        total = TP + TN + FP + FN
        acc = (TP + TN) / total if total else 0.0

        tpr = TP / (TP + FN) if (TP + FN) else 0.0
        tnr = TN / (TN + FP) if (TN + FP) else 0.0

        # precision / recall / F1
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = tpr
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        # advantage
        advantage = 2 * (acc - 0.5)

        print(
            f"[{split}] {k}_shots | "
            f"TPR={tpr:.4f} | "
            f"TNR={tnr:.4f} | "
            f"ACC={acc:.4f} | "
            f"F1={f1:.4f} | "
            f"ADV={advantage:.4f}"
        )

analyze("start")
print("-" * 40)
analyze("end")
