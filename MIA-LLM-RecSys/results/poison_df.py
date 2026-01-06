import argparse
import pickle
import numpy as np
import os

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

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    advantage = 2 * (acc - 0.5)

    return acc, advantage, f1


def find_best_threshold(member, nonmember, num_steps):
    thresholds = np.linspace(0, 1, num_steps)

    best = {
        "threshold": None,
        "accuracy": -1.0,
        "advantage": None,
        "f1": None,
    }

    for t in thresholds:
        acc, adv, f1 = compute_metrics(member, nonmember, t)

        if acc > best["accuracy"]:
            best["threshold"] = t
            best["accuracy"] = acc
            best["advantage"] = adv
            best["f1"] = f1

    return best

def main(args):
    print("=" * 80)
    print(
        f"Dataset={args.dataset} | Model={args.model} | Poison={args.poison_num}"
    )
    print("=" * 80)

    for k in args.k_shots:
        print(f"\n===== k = {k} shots =====")

        for pos in args.positions:
            member, nonmember = load_data(
                args.base_dir,
                args.dataset,
                args.poison_num,
                args.model,
                pos,
                k,
            )

            best = find_best_threshold(
                member, nonmember, args.num_thresholds
            )

            print(
                f"[{pos.upper():5s}] "
                f"best_th={best['threshold']:.4f} | "
                f"ACC={best['accuracy']:.4f} | "
                f"ADV={best['advantage']:.4f} | "
                f"F1={best['f1']:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default="./poison")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--poison_num", type=int, required=True)

    parser.add_argument(
        "--positions", nargs="+", default=["start", "end"]
    )
    parser.add_argument(
        "--k_shots", nargs="+", type=int, default=[1, 5, 10]
    )
    parser.add_argument(
        "--num_thresholds", type=int, default=1001
    )

    args = parser.parse_args()
    main(args)
