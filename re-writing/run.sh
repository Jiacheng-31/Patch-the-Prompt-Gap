export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python rewriting_vllm.py \
  --model-path ./grpo/grpo-lora-7b/mistral-step120 \
  --input-path ./dataset/NuminMathFilter/orginal_50k.jsonl \
  --output-path ./re-writing/mistral7b-grpo/50k-full.json \
  --success-output-path ./re-writing/mistral7b-grpo/50k-success.json \
  --tp-size 4 \
  --batch-size 36 \
  --max-new-tokens 8192 \
  --max-retries 0 \
  > ./re-writing/mistral7b-grpo/mistral7b.log 2>&1 &