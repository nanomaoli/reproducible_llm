export NCCL_P2P_DISABLE=1

# Assume running on a node with 8 A100 GPUs

# [Qwen, DeepSeek-Qwen], [All Datasets]
(export CUDA_VISIBLE_DEVICES=0,1,2,3
for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            python vllm_main.py --model Qwen/Qwen2.5-7B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            python vllm_main.py --model Qwen/Qwen2.5-7B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
        done
    done
done) &

# [Llama, DeepSeek-Llama], [All Datasets]
(export CUDA_VISIBLE_DEVICES=4,5,6,7
for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            python vllm_main.py --model meta-llama/Llama-3.1-8B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            python vllm_main.py --model meta-llama/Llama-3.1-8B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
        done
    done
done) &

wait