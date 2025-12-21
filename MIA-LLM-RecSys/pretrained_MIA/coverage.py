import pickle
import argparse

def main(path, dataset, model):
    with open(path+dataset+model+'/member.pkl', "rb") as file:
        member_data = pickle.load(file)
        # Count 1s in member_data
        member_ones = sum(1 for x in member_data if x == 1)
        member_total = len(member_data)
        member_ratio = member_ones / member_total if member_total > 0 else 0

        # Print results
        print(f"Member Set : 1s = {member_ones}, Total = {member_total}, Ratio = {member_ratio:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type =str, default="./results/pretrained/", help="output result path")
    parser.add_argument("--dataset", type=str, default="beauty", help="dataset")
    parser.add_argument("--model", type =str, default="/gpt-oss:20b", help="used LLM model")
    args = parser.parse_args()

    main(args.path, args.dataset, args.model)



