## Overview

This repository provides an experimental framework for studying **privacy risks in LLM-based recommender systems (RecSys)** under the **in-context learning (ICL)** paradigm.  
We focus on **membership inference attacks (MIAs)** that aim to determine whether a specific user–item interaction has appeared in the training data of the underlying large language model.

All attacks are conducted in a **black-box setting**, where the adversary only has access to model outputs via an API and does not rely on model parameters, gradients, or internal states.

The implemented attack strategies include:

1. **Inquiry-based Attack**
2. **Similarity-based Attack**
3. **Memorization-based Attack**
4. **Poisoning-based Attack**

The framework is **model-agnostic** and can be applied to any LLM that supports prompt-based inference.

---

## Supported Language Models

All language models are accessed **via the Ollama API**, ensuring reproducibility without distributing proprietary or large model weights.

The following models are currently supported:

- **LLaMA3-8B**
- **LLaMA4-Scout**
- **Mistral-7B**
- **Gemma3-4B**
- **GPT-OSS (20B)**
- **GPT-OSS (120B)**

Model selection is controlled through command-line arguments.  
No model weights are included in this repository.

> **Note:** The framework can be extended to additional models supported by Ollama with minimal modification.

---

## Dataset

Due to storage constraints, we provide the **MovieLens-1M (ml1m)** dataset in this repository.

- MovieLens-1M is sufficient to reproduce all **main experimental results** reported in the paper.
- The dataset is preprocessed into an **ICL-compatible recommendation format**, where user interaction histories are embedded into prompts.

The codebase is designed to support additional datasets (e.g., **Amazon Books** and **Amazon Beauty**) by following the same dataset interface.  
For our experiments, we additionally evaluate on the **Amazon Books** and **Amazon Beauty** datasets, which are publicly available from the Amazon Review corpus.

In Poisoning attack, we use IMDB for MovieLens-1M, you can download IMDB from its official website.
---

## Experimental Design

Our evaluation covers a total of **200 experimental configurations**, obtained by varying:

- the target LLM (**6 models**),
- the number of in-context demonstrations (**3 shot settings**),
- the position of injected examples in the prompt (**5 positions**), and
- the poisoning strategy (**3 variants**).

This design enables a systematic analysis of how **prompt composition**, **attack strategy**, and **model choice** jointly influence membership inference vulnerability.

---

## Running Attack Experiments

### Inquiry-based Attack

The inquiry-based membership inference attack can be executed using:

```bash
python3 inquiry.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end
````

---

### Similarity-based Attack

The similarity-based membership inference attack can be executed using:

```bash
python3 semantic.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end
```

---

### Memorization-based Attack

The memorization-based membership inference attack can be executed using:

```bash
python3 repeat.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end
```

---

### Poisoning-based Attack

The poisoning-based membership inference attack can be executed using:

```bash
python3 repeat.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end \
  --poison_num 1
```

The `--poison_num` argument controls the number of poisoned interactions injected into the prompt.

---

## Running Defense Experiments

Defense experiments follow the same command structure as attack experiments, with the attack scripts replaced by their corresponding defense implementations.

### Memorization Defense

```bash
cd Defense
python3 repeat_defense.py \
  --models llama3 \
  --datasets ml1m \
  --num_seeds 100 \
  --all_shots 1 \
  --positions end
```

### Inquiry Defense

Similarly, the inquiry-based defense can be executed by replacing `inquiry.py` with `inquiry_defense.py`.

> **Important:** For poisoning defenses, the `--poison_num` argument must still be specified.

---

## Reproducibility Notes

* All experiments are controlled using fixed random seeds.
* For convenience, we provide the complete experimental results for **GPT-OSS (120B)** in a single spreadsheet file, `Experimental_Results-GPT-OSS_120B.xlsx`, located in the `results/` directory.

