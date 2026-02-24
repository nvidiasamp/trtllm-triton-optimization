trtllm-build \
    --checkpoint_dir /workspace/artifacts/checkpoints/sq_checkpoint_Llama3.1-8B-Instruct \
    --output_dir /workspace/artifacts/engines/sq_engine_Llama3.1-8B-Instruct \
    --gpt_attention_plugin auto \
    --context_fmha enable \
    --gemm_plugin auto \
    --use_fused_mlp enable \
    --norm_quant_fusion enable \
    --reduce_fusion enable \