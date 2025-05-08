# Introduction

This project investigates the phenomenon where large language models (LLMs) return divergent responses for the same request. Heartfelt thanks for our collaborators' support. :saluting_face: 

## Environment Setup

#### 1. Software Packages
```bash
conda create -n reproducible_llm python=3.12 -y
conda activate reproducible_llm
pip install vllm
pip install datasets latex2sympy2 word2number immutabledict nltk langdetect
```

#### 2. Set up access to gated model/dataset:
```bash
export HF_TOKEN=token_text
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
```

## Steps to Launch Experiments

1. `git clone` this repository.
2. `cd reproducible_llm` navigate to the repository directory.
3. [Optional and Recommended] Run experiments in `tmux` terminals as they are long tasks.
4. Experiments are arranged in two scripts, supposing using 2 8-A100 nodes:
    - On the 1st node: `bash sh/run_node_1.sh`. This runs 4-GPU settings on 2 groups of 4 GPUs in parallel.
    - On the 2nd node: `bash sh/run_node_2.sh`. This runs 2-GPU settings on 4 groups of 2 GPUs in parallel.

Output files will be put under `./outputs/vllm` directory.

## Follow-Up Experiments

1. Experiments are arranged in 3 scripts, supposing using 1 8-A100 node:
    - These three scripts are using different GPUs, so, please run them Concurrently from 3 tmux sessions.
    - In any tmux session, `cd` to the root directory of this repo. Make sure you are using the reproducible_llm environment.
    - In the 1st tmux session: `bash sh/run_followup_1.sh`. This runs 4-GPU settings on GPU 4,5,6,7.
    - In the 2nd tmux sessionL `bash sh/run_followup_2.sh`. This runs some 2-GPU settings on GPU 0,1.
    - In the 3nd tmux sessionL `bash sh/run_followup_3.sh`. This runs some 2-GPU settings on GPU 2,3.

2. Output files will be archived and uploaded to a hf repo automatically, while locally they will be put under `./followup_exp_outputs/vllm` directory. 

3. For the reversed order experiments:
    - Also three scripts using different GPUs, so, please run them Concurrently from 3 tmux sessions.
    - In any tmux session, `cd` to the root directory of this repo. Make sure you are using the reproducible_llm environment.
    - In the 1st tmux session: `bash sh/run_reorder_1.sh`. This runs 4-GPU settings on GPU 4,5,6,7.
    - In the 2nd tmux sessionL `bash sh/run_reorder_2.sh`. This runs some 2-GPU settings on GPU 0,1.
    - In the 3nd tmux sessionL `bash sh/run_reorder_3.sh`. This runs some 2-GPU settings on GPU 2,3.