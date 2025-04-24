export NCCL_P2P_DISABLE=1

# Assume running on a node with 8 A100 GPUs

# [Qwen, DeepSeek-Qwen], [All Datasets]
(export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "$(date): START Group 1 on GPUs 0,1,2,3: [Qwen, DeepSeek-Qwen], [All Datasets]"

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model Qwen/Qwen2.5-7B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model Qwen/Qwen2.5-7B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_echo.log
        done
    done
done
echo "$(date): FINISH Group 1 on GPUs 0,1,2,3: [Qwen, DeepSeek-Qwen], [All Datasets]") &

# [Llama, DeepSeek-Llama], [All Datasets]
(export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "$(date): Starting Group 2 on GPUs 4,5,6,7: [Llama, DeepSeek-Llama], [All Datasets]"

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model meta-llama/Llama-3.1-8B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model meta-llama/Llama-3.1-8B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_${task}_${dtype}_bs_${batch_size} > 4A100_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_echo.log
        done
    done
done
echo "$(date): FINISH Group 2 on GPUs 4,5,6,7: [Llama, DeepSeek-Llama], [All Datasets]") &

wait

echo "$(date): FINISH 4A100 experiments" >> 4A100_echo.log