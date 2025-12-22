import pandas as pd
import json
import pickle
import numpy as np
import os
from copy import deepcopy

def load_ml1m():
    prompt_sentences = []
    with open(f"../processed_ml-1m/prompt_ml1m.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences

def load_beauty():
    prompt_sentences = []
    with open(f"../Beauty/prompt_Beauty.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences

def load_book():
    prompt_sentences = []
    with open(f"../Book/prompt_Book.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences

def load_pretrained_dataset(params):
    if params["dataset"] == "ml1m":
        prompt_sentences = load_ml1m()
        params["prompt_prefix"] = (
            "You are a recommender system. "
            "You will receive user–item interaction set derived from the MovieLens-1M dataset. "
            "Your task is to answer only \"Yes\" or \"No\", indicating whether you have previously seen the user-item interaction set. "
            "No additional text is allowed.\n\n"
        )
        params["task_format"] = "recommendation"
    elif params["dataset"] == "beauty":
        prompt_sentences = load_beauty()
        params[
            "prompt_prefix"
        ] = (
            "You are a recommender system. "
            "You will receive user–item interaction set derived from the Amazon Beauty dataset. "
            "Your task is to answer only \"Yes\" or \"No\", indicating whether you have previously seen the user-item interaction set. "
            "No additional text is allowed.\n\n"
        )
        params["task_format"] = "recommendation"
    elif params["dataset"] == "book":
        prompt_sentences = load_book()
        params[
            "prompt_prefix"
        ] = (
            "You are a recommender system. "
            "You will receive user–item interaction set derived from the Amazon Book dataset. "
            "Your task is to answer only \"Yes\" or \"No\", indicating whether you have previously seen the user-item interaction set. "
            "No additional text is allowed.\n\n"
        )
        params["task_format"] = "recommendation"
    else:
        raise NotImplementedError
    return prompt_sentences

if __name__ == "__main__":
    params = {}
    params["dataset"] = "ml1m"
    load_pretrained_dataset(params)
    print(params)
