# trtllm-triton-optimization

Optimization and deployment workflows for LLM inference using TensorRT-LLM and Triton Inference Server.

本READMEでは、以下の流れを説明します：

1. Docker Compose で環境起動  
2. Hugging Face からモデル取得  
3. TensorRT-LLM でエンジンビルド  
4. Triton Inference Server にデプロイ  
5. （おまけ）精度・スループット評価

---

## 前提条件

- NVIDIA GPU（L40）
- NVIDIA Driver インストール済み
- Docker + Docker Compose
- NVIDIA Container Toolkit
- Hugging Face アクセストークン取得済み

### 確認コマンド

```bash
nvidia-smi
docker --version
docker compose version
```

---

## ディレクトリ構成

```text
project/
├── docker/
├── workspace/
│   ├── scripts/
│   │   ├── checkpoints/
│   │   └── engines/
│   └── artifacts/
│       ├── models/
│       ├── checkpoints/
│       ├── engines/
│       └── model_repository/
└── README.md
```

---

# 1. Docker Compose 起動

```bash
cd ./workspace
docker compose up -d
docker exec -it triton_server bash
```

---

# 2. モデル取得（例：Meta-Llama-3.1-8B-Instruct）

事前にログイン：

```bash
hf auth login
```

モデル取得：

```bash
cd /workspace/artifacts/models
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

---

# 3. TensorRT-LLM エンジン作成

## 3-1. チェックポイント変換（AWQ量子化）

```bash
python3 /app/examples/quantization/quantize.py \
  --model_dir /workspace/artifacts/models/Meta-Llama-3.1-8B-Instruct \
  --output_dir /workspace/artifacts/checkpoints/awq_checkpoint_Llama3.1-8B-Instruct \
  --dtype float16 \
  --qformat int4_awq \
  --awq_block_size 128 \
  --calib_size 32 \
  --kv_cache_dtype int8
```

または：

```bash
bash /workspace/scripts/checkpoints/awq_checkpoint.bash
```

---

## 3-2. エンジンビルド

```bash
trtllm-build \
  --checkpoint_dir /workspace/artifacts/checkpoints/awq_checkpoint_Llama3.1-8B-Instruct \
  --output_dir /workspace/artifacts/engines/awq_engine_Llama3.1-8B-Instruct \
  --gpt_attention_plugin auto \
  --context_fmha enable \
  --gemm_plugin auto \
  --use_fused_mlp enable \
  --norm_quant_fusion enable \
  --reduce_fusion enable
```

または：

```bash
bash /workspace/scripts/engines/awq_engine.bash
```

---

# 4. Triton Inference Server にデプロイ

モデルテンプレートコピー：

```bash
cp -r /app/all_models/inflight_batcher_llm/* \
      /workspace/artifacts/model_repository/
```

---

## 4-1. config.pbtxt 作成

```bash
ENGINE_DIR=/workspace/artifacts/engines/awq_engine_Llama-3.1-8B-Instruct
TOKENIZER_DIR=/workspace/artifacts/models/Meta-Llama-3.1-8B-Instruct
MODEL_FOLDER=/workspace/artifacts/model_repository
TRITON_MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=2000
MAX_QUEUE_SIZE=32
FILL_TEMPLATE_SCRIPT=/app/tools/fill_template.py
DECOUPLED_MODE=false
LOGITS_DATATYPE=TYPE_FP32
```

```bash
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt \
  triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt \
  tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt \
  triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE}

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt \
  tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt \
  triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}
```

または：

```bash
bash /workspace/scripts/make_config.bash
```

---

## 4-2. Triton Inference Server 起動

```bash
python /app/scripts/launch_triton_server.py \
  --world_size=1 \
  --model_repo=/workspace/artifacts/model_repository
```

---

## 4-3. 推論テスト（localhost）

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "What is machine learning?",
    "max_tokens": 24,
    "bad_words": "",
    "stop_words": ""
  }'
```

---

# 5. 精度・スループット評価

精度評価

```bash
trtllm-eval \
  --model /workspace/artifacts/engines/sq_engine_Llama3.1-8B-Instruct \
  --tokenizer /workspace/artifacts/models/Llama-3.1-8B-Instruct \
  --backend tensorrt mmlu
```

または：

```bash
bash /workspace/scripts/evaluate_accuracy.bash
```

---

スループット評価

```bash
trtllm-bench \
  --model /workspace/artifacts/models/Llama-3.1-8B-Instruct  throughput \
  --engine_dir /workspace/artifacts/engines/awq_engine_Llama3.1-8B-Instruct \
  --dataset /workspace/artifacts/synthetic_1024_1024.txt \
  --concurrency 32 \
  --backend tensorrt
```

または：

```bash
bash /workspace/scripts/benchmark_throughput.bash
```