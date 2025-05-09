export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3


for task in livecodebench_easy; do
    for dtype in bfloat16; do
        for batch_size in 32; do
            echo "$(date): START CATCHUP experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
            python vllm_main_followup_exp.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_CATCHUP_${task}_${dtype}_bs_${batch_size} > 2A100_CATCHUP_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
        done
    done
done

for task in math500 gpqa_diamond; do
    for dtype in float32; do
        for batch_size in 32 16 8; do
            echo "$(date): START experiment: LAYERCAST, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
            python vllm_layercast.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                --task $task \
                --dtype $dtype \
                --seed 42 \
                --batch_size $batch_size \
                --max_tokens 32768 \
                --exp_name 2A100_LAYERCAST_${task}_${dtype}_bs_${batch_size} > 2A100_LAYERCAST_deepseek-qwen_${task}_${dtype}_bs_${batch_size}.log 2>&1
            echo "$(date): FINISH experiment: LAYERCAST, Model=DeepSeek-Qwen, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
        done
    done
done

for task in aime24; do
    for dtype in bfloat16 float16 float32; do
        for batch_size in 32 16 8; do
            for passk in 16; do
                echo "$(date): START experiment: PASS_${passk}, Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
                python vllm_passk.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 32768 \
                    --passk $passk \
                    --exp_name 2A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > 2A100_PASS_${passk}_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
            done
        done
    done
done

for task in math500; do
    for dtype in bfloat16 float16 float32; do
        for batch_size in 32 16 8; do
            for passk in 4; do
                echo "$(date): START experiment: PASS_${passk}, Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
                python vllm_passk.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 32768 \
                    --passk $passk \
                    --exp_name 2A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > 2A100_PASS_${passk}_deepseek-llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=DeepSeek-Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
            done
        done
    done
done

for task in aime24; do
    for dtype in bfloat16 float16 float32; do
        for batch_size in 32 16 8; do
            for passk in 16; do
                echo "$(date): START experiment: PASS_${passk}, Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
                python vllm_passk.py --model meta-llama/Llama-3.1-8B-Instruct \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 2048 \
                    --passk $passk \
                    --exp_name 2A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > 2A100_PASS_${passk}_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
            done
        done
    done
done

for task in math500; do
    for dtype in bfloat16 float16 float32; do
        for batch_size in 32 16 8; do
            for passk in 4; do
                echo "$(date): START experiment: PASS_${passk}, Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
                python vllm_passk.py --model meta-llama/Llama-3.1-8B-Instruct \
                    --task $task \
                    --dtype $dtype \
                    --seed 42 \
                    --batch_size $batch_size \
                    --max_tokens 2048 \
                    --passk $passk \
                    --exp_name 2A100_PASS_${passk}_${task}_${dtype}_bs_${batch_size} > 2A100_PASS_${passk}_llama_${task}_${dtype}_bs_${batch_size}.log 2>&1
                echo "$(date): FINISH experiment: PASS_${passk}, Model=Llama, Task=$task, Dtype=$dtype, Batch Size=$batch_size" >> 2A100_followup.log
            done
        done
    done
done