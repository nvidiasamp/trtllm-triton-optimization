python3 /app/examples/quantization/quantize.py \
    --model_dir /workspace/artifacts/models/Llama-3.1-8B-Instruct \
    --output_dir /workspace/artifacts/checkpoints/sq_checkpoint_Llama3.1-8B-Instruct \
    --dtype float16 \
    --qformat int8_sq \
    --kv_cache_dtype int8 \