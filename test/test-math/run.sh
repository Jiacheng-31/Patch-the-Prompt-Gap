export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u eval_vllm.py \
  --model-path ./output/Mistral-7B-data-agent-full \
  --data-dir ./test-math/dataset \
  --files \
    AGIEval_sat-math_orca.jsonl \
    amc23_orca.jsonl \
    IMO_numeric_bench.jsonl \
    math500_orca.jsonl \
    minervamath_test_orca.jsonl \
  --tp-size 4 \
  --batch-size 16 \
  --max-new-tokens 4096 \
  --output-dir ./7B-data-agent-full \
  2>&1 | tee ./output_log/7B-data-agent-full.log
