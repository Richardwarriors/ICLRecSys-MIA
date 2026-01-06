import argparse
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(base_dir, dataset, poison_num, model, position, k):
    path = os.path.join(
        base_dir,
        dataset,
        f"{poison_num}",
        model,
        position,
        f"{k}_shots",
    )

    member_path = os.path.join(path, "member.pkl")
    nonmember_path = os.path.join(path, "nonmember.pkl")

    if not os.path.exists(member_path) or not os.path.exists(nonmember_path):
        raise FileNotFoundError(f"Missing files in {path}")

    with open(member_path, "rb") as f:
        member = pickle.load(f)

    with open(nonmember_path, "rb") as f:
        nonmember = pickle.load(f)

    return np.array(member), np.array(nonmember)


def compute_metrics(member, nonmember, threshold):
    TP = np.sum(member >= threshold)
    FN = np.sum(member < threshold)
    TN = np.sum(nonmember < threshold)
    FP = np.sum(nonmember >= threshold)

    acc = (TP + TN) / (TP + TN + FP + FN)
    advantage = 2 * (acc - 0.5)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return acc, advantage, f1



def sweep_thresholds(member, nonmember, num_steps):
    thresholds = np.linspace(0, 1, num_steps)

    adv_curve = []
    best = {
        "threshold": None,
        "accuracy": -1.0,
        "advantage": None,
        "f1": None,
    }

    for t in thresholds:
        acc, adv, f1 = compute_metrics(member, nonmember, t)
        adv_curve.append(adv)

        if acc > best["accuracy"]:
            best["threshold"] = t
            best["accuracy"] = acc
            best["advantage"] = adv
            best["f1"] = f1

    return thresholds, np.array(adv_curve), best



def plot_poisoning_adv_curve(thresholds, adv_dict, position, save_path):
    plt.figure(figsize=(4.5, 3.2))  

    for k, adv in adv_dict.items():
        plt.plot(
            thresholds,
            adv,
            linewidth=2,
            label=f"{k}-shot",
        )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Attack Advantage")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")

def main(args):
    print("=" * 80)
    print(
        f"Dataset={args.dataset} | Model={args.model} | Poison={args.poison_num}"
    )
    print("=" * 80)

    for pos in args.positions:
        print(f"\n===== POSITION = {pos.upper()} =====")

        adv_curves = {}

        for k in args.k_shots:
            member, nonmember = load_data(
                args.base_dir,
                args.dataset,
                args.poison_num,
                args.model,
                pos,
                k,
            )

            thresholds, adv_curve, best = sweep_thresholds(
                member, nonmember, args.num_thresholds
            )

            adv_curves[k] = adv_curve

            print(
                f"[{pos.upper():5s} | {k}-shot] "
                f"best_th={best['threshold']:.4f} | "
                f"ACC={best['accuracy']:.4f} | "
                f"ADV={best['advantage']:.4f} | "
                f"F1={best['f1']:.4f}"
            )

        save_path = os.path.join(
            args.base_dir,
            args.dataset,
            f"poison_{args.poison_num}_{args.model}_{pos}_k_comparison.png",
        )

        plot_poisoning_adv_curve(
            thresholds,
            adv_curves,
            pos,
            save_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default="./poison")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--poison_num", type=int, required=True)

    parser.add_argument(
        "--positions", nargs="+", default=["start"]
    )
    parser.add_argument(
        "--k_shots", nargs="+", type=int, default=[1, 5, 10]
    )
    parser.add_argument(
        "--num_thresholds", type=int, default=1001
    )

    args = parser.parse_args()
    main(args)
