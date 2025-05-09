export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "$(date): START Group 1 on GPUs 4,5,6,7: for 4-A100 experiments" >> 4A100_followup.log

for task in math500; do
    for dtype in float32 float16 bfloat16; do
        for batch_size in 8 16 32; do
            for passk in 4; do
                echo "$(date): START experiment: PASS_${passk}, Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
                python vllm_passk.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 32768 \
                    --passk $passk \
                    --exp_name 4A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > logs/4A100_PASS_${passk}_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            done
        done
    done
done

for task in aime24; do
    for dtype in float32 float16 bfloat16; do
        for batch_size in 8 16 32; do
            for passk in 16; do
                echo "$(date): START experiment: PASS_${passk}, Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
                python vllm_passk.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 32768 \
                    --passk $passk \
                    --exp_name 4A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > logs/4A100_PASS_${passk}_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            done
        done
    done
done

for task in math500; do
    for dtype in float32 float16 bfloat16; do
        for batch_size in 8 16 32; do
            for passk in 4; do
                echo "$(date): START experiment: PASS_${passk}, Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
                python vllm_passk.py --model meta-llama/Llama-3.1-8B-Instruct \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 2048 \
                    --passk $passk \
                    --exp_name 4A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > logs/4A100_PASS_${passk}_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            done
        done
    done
done

for task in math500; do
    for dtype in float32 float16 bfloat16; do
        for batch_size in 8 16 32; do
            for passk in 4; do
                echo "$(date): START experiment: PASS_${passk}, Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
                python vllm_passk.py --model Qwen/Qwen2.5-7B-Instruct \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 2048 \
                    --passk $passk \
                    --exp_name 4A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > logs/4A100_PASS_${passk}_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            done
        done
    done
done

for task in aime24; do
    for dtype in float32 float16 bfloat16; do
        for batch_size in 8 16 32; do
            for passk in 16; do
                echo "$(date): START experiment: PASS_${passk}, Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
                python vllm_passk.py --model meta-llama/Llama-3.1-8B-Instruct \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 2048 \
                    --passk $passk \
                    --exp_name 4A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > logs/4A100_PASS_${passk}_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            done
        done
    done
done

for task in aime24; do
    for dtype in float32 float16 bfloat16; do
        for batch_size in 8 16 32; do
            for passk in 16; do
                echo "$(date): START experiment: PASS_${passk}, Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
                python vllm_passk.py --model Qwen/Qwen2.5-7B-Instruct \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 2048 \
                    --passk $passk \
                    --exp_name 4A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > logs/4A100_PASS_${passk}_qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            done
        done
    done
done

for task in gpqa_diamond livecodebench_hard livecodebench_medium livecodebench_easy; do
    for dtype in float32; do
        for batch_size in 8 16 32; do
            echo "$(date): START experiment: LAYERCAST, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            python vllm_layercast.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_LAYERCAST_${task}_${dtype}_bs_${batch_size} > logs/4A100_LAYERCAST_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: LAYERCAST, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
        done
    done
done

for task in math500; do
    for dtype in float32 float16 bfloat16; do
        for batch_size in 8 16 32; do
            for passk in 4; do
                echo "$(date): START experiment: PASS_${passk}, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
                python vllm_passk.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 32768 \
                    --passk $passk \
                    --exp_name 4A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > logs/4A100_PASS_${passk}_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            done
        done
    done
done

for task in aime24; do
    for dtype in float32 float16 bfloat16; do
        for batch_size in 8 16 32; do
            for passk in 16; do
                echo "$(date): START experiment: PASS_${passk}, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
                python vllm_passk.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 32768 \
                    --passk $passk \
                    --exp_name 4A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > logs/4A100_PASS_${passk}_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            done
        done
    done
done

for task in math500 aime24; do
    for dtype in float32; do
        for batch_size in 8 16 32; do
            echo "$(date): START experiment: LAYERCAST, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
            python vllm_layercast.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 4A100_LAYERCAST_${task}_${dtype}_bs_${batch_size} > logs/4A100_LAYERCAST_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: LAYERCAST, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 4A100_followup.log
        done
    done
done

########################################################################################
###### This is the old beginning of the script, We will reverse the order of the tasks in the script



echo "$(date): FINISH Group 1 on GPUs 4,5,6,7: for 4-A100 experiments" >> 4A100_followup.log


