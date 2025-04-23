# Introduction

This project investigates the phenomenon where large language models (LLMs) return divergent responses for the same request. Heartfelt thanks for our collaborators' support. :saluting_face: 

## Environment Setup

```bash
conda create -n reproducible_llm python=3.12 -y
conda activate reproducible_llm
pip install vllm
pip install datasets latex2sympy2 word2number immutabledict nltk langdetect
```

#### side note on gated model/dataset:
`meta-llama/Llama-3.1-8B-Instruct` is a **gated model** and `Idavidrein/gpqa` is a **gated dataset**. Please navigate to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and https://huggingface.co/datasets/Idavidrein/gpqa to request access for them, respectively. (There could be different online repos for a dataset, please use the link given here because it is the one used in our evaluation)

After access requests are approved, which usually happens shortly after your requests, make sure you login your machine with your HuggingFace account. [reference link](https://huggingface.co/docs/hub/en/models-gated#download-files)
```bash
huggingface-cli login
```

## Steps to Launch Experiments

1. `git clone` this repository.
2. `cd reproducible_llm` navigate to the repository directory.
3. [Optional and Recommended] Run experiments in tmux terminals as they are long tasks.
4. Experiments are arranged in two scripts, supposing using 2 8-A100 nodes:
    - On the 1st node: `bash sh/run_node_1.sh`. This runs 4-GPU settings on 2 groups of 4 GPUs in parallel.
    - On the 2nd node: `bash sh/run_node_2.sh`. This runs 2-GPU settings on 4 groups of 2 GPUs in parallel.

Output files will be put under `./outputs/vllm` directory.