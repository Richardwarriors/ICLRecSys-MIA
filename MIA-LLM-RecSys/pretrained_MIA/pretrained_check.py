import sys
import os
import argparse
import pickle
from copy import deepcopy
import requests
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    load_pretrained_dataset,
)

def main(models, datasets, num_seeds):
    default_params = {}
    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for seed in range(num_seeds):
                p = deepcopy(default_params)
                p["model"] = model
                p["dataset"] = dataset
                p["seed"] = seed
                p["expr_name"] = f"{p['dataset']}_{p['model']}_subsample_seed{p['seed']}"
                all_params.append(p)

    all_member_list = []


    for param_index, params in enumerate(all_params):
        prompt_subset = prepare_data(params)
        print(len(prompt_subset))


        target_sentence = prompt_subset[params['seed']]
        required_for_mem = inquiry(params, target_sentence)
        if required_for_mem is None:
            continue

        all_member_list.append(required_for_mem)

        save_path = f"./results/pretrained/{params['dataset']}/{params['model']}/"
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'member.pkl'), "wb") as file:
            pickle.dump(all_member_list, file)

def prepare_data(params):
    print("\nExperiment name:", params["expr_name"])
    prompted_sentences = load_pretrained_dataset(params)
    return prompted_sentences

def inquiry(params, test_sentence):
    query_sentence = test_sentence
    #print(f"query_sentence: {query_sentence}")
    input_to_model = construct_prompt_omit(params)
    print(f"input_to_model: {input_to_model}")
    print(f"query_sentence: {query_sentence}")
    #return_idx = query_ollama(input_to_model, params['model'])
    return_idx = query_ollama_chat(input_to_model,  query_sentence, params['model'])
    return return_idx

def query_ollama_chat(prompt_setup, prompt_question, model, max_token = 2, temperature=0.0):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt_setup},
            {"role": "user", "content": prompt_question}
        ],
        "max_tokens": max_token,
        "temperature": temperature,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        raw_output = data.get("message", {}).get("content", "").strip().lower()
        print(f"[Ollama chat output]: {raw_output}")

        raw_output = raw_output.split()[0].strip(",.?!")
        #print(f"[Ollama chat output]: {raw_output}")

        if raw_output.startswith("yes"):
            return 1
        elif raw_output.startswith("no"):
            return 0
        else:
            print("[Warning] Unexpected response format.")
            return -1

    except requests.RequestException as e:
        print(f"[Error] Request to Ollama chat API failed: {e}")
        return -1

def construct_prompt_omit(params):
    prompt = params.get("prompt_prefix", "")

    return prompt

def convert_to_list(items, is_int=False):
    return [int(s.strip()) if is_int else s.strip() for s in items.split(",")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--num_seeds", type=int, required=True)

    args = vars(parser.parse_args())
    args["models"] = convert_to_list(args["models"])
    args["datasets"] = convert_to_list(args["datasets"])
    print("DEBUG: args = ", args)
    main(**args)
