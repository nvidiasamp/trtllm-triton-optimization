python3 /app/examples/quantization/quantize.py \
    --model_dir /workspace/artifacts/models/Llama-3.1-8B-Instruct \
    --output_dir /workspace/artifacts/checkpoints/awq_checkpoint_Llama3.1-8B-Instruct \
    --dtype float16 \
    --qformat int4_awq \
    --awq_block_size 128 \
    --calib_size 32 \
    --kv_cache_dtype int8 \