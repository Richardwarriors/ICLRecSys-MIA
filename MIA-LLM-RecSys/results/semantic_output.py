import pickle
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-oss:20b")
parser.add_argument("--dataset", type=str, default="ml1m")
args = parser.parse_args()

MODEL = args.model
DATASET = args.dataset
BASE_DIR = Path(f"./semantic/{DATASET}/{MODEL}")

k_values = [1, 5, 10]
thresholds = np.linspace(0.0, 1.0, 200)  


def plot_adv_curve(thresholds, adv_curves, split):
    plt.figure(figsize=(4.5, 3.2))  
    for k, adv in adv_curves.items():
        plt.plot(
            thresholds,
            adv,
            linewidth=2,
            label=f"{k}-shot",
        )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Attack Advantage")
    plt.ylim(0, 0.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)

    plt.tight_layout()
    out_path = BASE_DIR / f"semantic_{MODEL}_{split}_adv_curve.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out_path}")



def analyze(split: str):
    print(f"\n===== {split.upper()} =====")

    adv_curves = {}

    for k in k_values:

        with open(BASE_DIR / split / f"{k}_shots" / "member.pkl", "rb") as f:
            member_data = np.array(pickle.load(f))

        with open(BASE_DIR / split / f"{k}_shots" / "nonmember.pkl", "rb") as f:
            nonmember_data = np.array(pickle.load(f))

        adv_curve = []

        best_acc = -1.0
        best_th = None
        best_f1 = 0.0
        best_adv = 0.0
        best_tpr = 0.0
        best_tnr = 0.0

        for threshold in thresholds:
            TP = np.sum(member_data >= threshold)
            FN = np.sum(member_data < threshold)

            TN = np.sum(nonmember_data < threshold)
            FP = np.sum(nonmember_data >= threshold)

            acc = (TP + TN) / (TP + TN + FP + FN)

            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = tpr

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            advantage = 2 * (acc - 0.5)
            adv_curve.append(advantage)

            if acc > best_acc:
                best_acc = acc
                best_th = threshold
                best_f1 = f1
                best_adv = advantage
                best_tpr = tpr
                best_tnr = tnr

        adv_curves[k] = np.array(adv_curve)

        print(
            f"{k}_shots | "
            f"best_th={best_th:.4f} | "
            f"TPR={best_tpr:.4f} | "
            f"TNR={best_tnr:.4f} | "
            f"F1={best_f1:.4f} | "
            f"Advantage={best_adv:.4f}"
        )

    plot_adv_curve(thresholds, adv_curves, split)


if __name__ == "__main__":
    analyze("start")
    analyze("end")
