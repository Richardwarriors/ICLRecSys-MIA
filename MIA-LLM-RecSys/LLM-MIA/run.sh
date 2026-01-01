#!/bin/bash

mkdir -p logs

# 1. Define models, datasets, poison numbers
models=("llama3" "gemma3:4b" "mistral:7b" "gpt-oss:20b" "gpt-oss:120b")
#models=("mistral:7b" "gpt-oss:20b")
datasets=("ml1m")
poison_nums=(5)

for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    for poison_num in "${poison_nums[@]}"
    do
      for pos in start
      do
        for shot in 1 
        do
          echo "Running model=$model, dataset=$dataset, poison_num=$poison_num, pos=$pos, shots=$shot..."
          START=$(date +%s)

          python3 poisoning.py \
            --models "$model" \
            --datasets "$dataset" \
            --num_seeds 100 \
            --all_shots "$shot" \
            --positions "$pos" \
            --poison_num "$poison_num" \
            > logs/${model//:/_}_${dataset}_poison${poison_num}_shot${shot}_${pos}.log 2>&1

          END=$(date +%s)
          echo "Finished model=$model, dataset=$dataset, poison_num=$poison_num in $((END-START)) seconds."
        done
      done
    done
  done
done
