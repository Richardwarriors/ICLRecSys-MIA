import pickle
import argparse

def main(path, dataset, model):
    with open(path+dataset+'/'+model+'/member.pkl', "rb") as file:
        member_data = pickle.load(file)

    with open(f"../pretrained_RecSys/{dataset}_test.txt", "r") as f:
        lines = f.readlines()

    # 3. 安全检查（防止长度不一致 silent bug）
    assert len(member_data) == len(lines), (
        f"Length mismatch: member_data={len(member_data)}, lines={len(lines)}"
    )

    res = 0

    for i in range(len(member_data)):
        member_item = str(member_data[i]).strip()
        txt_item = lines[i].strip()

        if member_item == txt_item:
            res += 1

    accuracy = res / len(member_data)

    print(f"Match count: {res}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type =str, default="./results/pretrained/", help="output result path")
    parser.add_argument("--dataset", type=str, default="ml1m", help="dataset")
    parser.add_argument("--model", type =str, default="llama3", help="used LLM model")
    args = parser.parse_args()

    main(args.path, args.dataset, args.model)



