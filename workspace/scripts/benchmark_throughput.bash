trtllm-bench \
    --model /workspace/artifacts/models/Llama-3.1-8B-Instruct  throughput \
    --engine_dir /workspace/artifacts/engines/awq_engine_Llama3.1-8B-Instruct \
    --dataset /workspace/artifacts/synthetic_1024_1024.txt \
    --concurrency 32 \
    --backend tensorrt