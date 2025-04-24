export NCCL_P2P_DISABLE=1

# Assume running on a node with 8 A100 GPUs

# [Qwen, DeepSeek-Qwen], [aime24, livecodebench_easy, livecodebench_medium, livecodebench_hard]
(export CUDA_VISIBLE_DEVICES=0,1
echo "$(date): START Group 1 on GPUs 0,1: [Qwen, DeepSeek-Qwen], [aime24, livecodebench_easy, livecodebench_medium, livecodebench_hard]"

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model Qwen/Qwen2.5-7B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model Qwen/Qwen2.5-7B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done
echo "$(date): FINISH Group 1 on GPUs 0,1: [Qwen, DeepSeek-Qwen], [aime24, livecodebench_easy, livecodebench_medium, livecodebench_hard]") &

# [Llama, DeepSeek-Llama], [aime24, livecodebench_easy, livecodebench_medium, livecodebench_hard]
(export CUDA_VISIBLE_DEVICES=2,3
echo "$(date): START Group 2 on GPUs 2,3: [Llama, DeepSeek-Llama], [aime24, livecodebench_easy, livecodebench_medium, livecodebench_hard]"

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model meta-llama/Llama-3.1-8B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model meta-llama/Llama-3.1-8B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in aime24 livecodebench_easy livecodebench_medium livecodebench_hard; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done
echo "$(date): FINISH Group 2 on GPUs 2,3: [Llama, DeepSeek-Llama], [aime24, livecodebench_easy, livecodebench_medium, livecodebench_hard]") &


# [Qwen, DeepSeek-Qwen], [math500, gpqa_diamond]
(export CUDA_VISIBLE_DEVICES=4,5
echo "$(date): Starting Group 3 on GPUs 4,5: [Qwen, DeepSeek-Qwen], [math500, gpqa_diamond]"

for task in math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model Qwen/Qwen2.5-7B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model Qwen/Qwen2.5-7B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done
echo "$(date): FINISH Group 3 on GPUs 4,5: [Qwen, DeepSeek-Qwen], [math500, gpqa_diamond]") &

# [Llama, DeepSeek-Llama], [math500, gpqa_diamond]
(export CUDA_VISIBLE_DEVICES=6,7
echo "$(date): Starting Group 4 on GPUs 6,7: [Llama, DeepSeek-Llama], [math500, gpqa_diamond]"

for task in math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model meta-llama/Llama-3.1-8B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in math500 gpqa_diamond; do
    for dtype in bfloat16 float16; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model meta-llama/Llama-3.1-8B-Instruct \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 2000 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done

for task in math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size"
            python vllm_main.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_${task}_${dtype}_bs_${batch_size} > 2A100_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_echo.log
        done
    done
done
echo "$(date): FINISH Group 4 on GPUs 6,7: [Llama, DeepSeek-Llama], [math500, gpqa_diamond]") &


wait

echo "$(date): FINISH 2A100 experiments" >> 2A100_echo.log