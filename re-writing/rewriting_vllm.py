import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

"""
脚本功能：
- 读取 jsonl 数学数据集（每行一个样本）
- 利用一个 LLM 重写每道题的解答 solution：
    * 保持推理正确
    * 保持最终 boxed 答案和原来一致
- 重写失败（答案对不上）时回退到原始 solution
- 输出两个 JSON（list）文件：
    1) 全量重写结果
    2) 只包含“重写成功”的子集
"""

# ====== boxed 答案的正则（和你别处保持一致，带不带 $ 都行） ======
BOX_PATTERN = re.compile(
    r"\$?\s*\\boxed\{(.*?)\}\s*\$?",
    re.DOTALL,
)


def extract_gold_answer(text: str | None) -> str | None:
    """
    从原始 solution 文本中抽取“标准答案”（boxed 里的内容）。
    - 优先：匹配 \\boxed{...}（外面 $ 可有可无）
    - 如果完全没有 boxed，则退化为取最后一个数字（比较脏，但保留原逻辑）
    返回：str（可能是 '42'，也可能是 '2122, 2123, 2124, 2125'），或 None
    """
    if text is None:
        return None

    m = BOX_PATTERN.search(text)
    if m:
        return m.group(1).strip()

    # 退化策略：找所有数字，取最后一个
    nums = re.findall(r"[\-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()

    return None


def extract_reasoning(text: str | None) -> str:
    """
    抽取“推理过程”文本：把含有 \\boxed 的行全部去掉。
    用于分析 / debug 时查看推理，本脚本目前不再用于 prompt。
    """
    if text is None:
        return ""
    lines = text.splitlines()
    # 这里注意：读进来的字符串里是 '\boxed'（一个反斜杠）
    kept = [ln for ln in lines if "\\boxed" not in ln]
    reasoning = "\n".join(kept).strip()
    return reasoning


def build_prompt(tokenizer, question: str, original_solution: str) -> str:
    """
    构造给 vLLM 的 prompt。
    这里是“重写型”prompt，不是你评测时的解题 prompt——
    目的是让模型改写 solution，同时：
      - 推理要正确
      - 最终答案不变
      - 最后一行输出 $\\boxed{...}$

    注意：这里不需要用 .format()，所以可以直接写 $\\boxed{56}$。
    """
    user_message = (
        "You are an expert in math word problems. Below is a math word problem and an existing step-by-step solution.\n"
        "Please rewrite the solution in your own words while keeping all reasoning steps correct "
        "and keeping the final answer the same.\n"
        "Follow these rules:\n"
        "- Keep the explanation clear and step by step.\n"
        "- Do NOT mention that you are rewriting another solution.\n"
        "- At the end, on a separate line, output ONLY the final answer "
        "in LaTeX boxed format, exactly like:\n"
        "$\\boxed{56}$\n"
        "Use dollar signs and \\boxed{} exactly as shown.\n\n"
        "Problem:\n"
        f"{question}\n\n"
        "Existing solution (for reference, do NOT copy it verbatim):\n"
        f"{original_solution}\n"
    )

    messages = [
        {"role": "user", "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,              # 返回字符串，由 vLLM 再去 tokenize
        add_generation_prompt=True,  # 自动加上 assistant 起始标记
    )
    return prompt


def extract_pred_answer_from_output(text: str | None) -> str | None:
    """
    从模型生成的文本中抽取 boxed 答案：
    - 优先：\\boxed{...}
    - 否则退化为最后一个数字（和上面的 gold 提取逻辑对齐）
    """
    if text is None:
        return None

    m = BOX_PATTERN.search(text)
    if m:
        return m.group(1).strip()

    nums = re.findall(r"[\-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()

    return None


# ====== 主逻辑：读取 jsonl，重写 solution，校验答案 ======

def rewrite_dataset(
    model_path: str,
    input_path: str,
    output_path: str,
    success_output_path: str | None = None,
    tp_size: int = 1,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    max_retries: int = 2,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if success_output_path is None:
        success_output_path = output_path.with_name(
            output_path.stem + "_rewrite_success" + output_path.suffix
        )
    success_output_path = Path(success_output_path)

    # 1. 读取“原始 jsonl 数据”
    data = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)

    print(f"Loaded dataset from {input_path}, total samples = {len(data)}")

    questions: list[str] = []
    gold_raw_answers: list[str] = []

    for ex in data:
        # ---- 题目：优先用 problem，没有就从 messages 里 user 拿 ----
        q = ex.get("problem")
        if not q:
            msgs = ex.get("messages") or []
            for m in msgs:
                if m.get("role") == "user":
                    q = m.get("content", "")
                    if q:
                        break
        if q is None:
            q = ""
        questions.append(q)

        # ---- 原始解答：优先用 solution，没有就从 messages 里 assistant 拿 ----
        a = ex.get("solution")
        if not a:
            msgs = ex.get("messages") or []
            for m in msgs:
                if m.get("role") == "assistant":
                    a = m.get("content", "")
                    if a:
                        break
        if a is None:
            a = ""
        gold_raw_answers.append(a)

    # 提前抽出：标准答案（boxed 内容） & 原始推理（仅用于记录）
    gold_answers = [extract_gold_answer(a) for a in gold_raw_answers]
    reasonings = [extract_reasoning(a) for a in gold_raw_answers]

    # 2. 初始化 tokenizer & vLLM 模型
    print(f"Initializing tokenizer & vLLM from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=None,
    )

    # 3. 逐 batch 重写
    rewritten_records = []
    success_records = []

    num_bad = 0

    for start in tqdm(range(0, len(data), batch_size), desc="Rewriting"):
        end = min(start + batch_size, len(data))
        batch_indices = list(range(start, end))

        # 当前 batch 中，每条样本的重写状态
        remaining_indices = batch_indices.copy()
        trial_counts: dict[int, int] = {idx: 0 for idx in batch_indices}
        final_rewritten: dict[int, str] = {}
        rewrite_success: dict[int, bool] = {idx: False for idx in batch_indices}

        while remaining_indices:
            prompts = []
            cur_indices = []

            # 构造这一轮要重写的样本的 prompt
            for idx in remaining_indices:
                q = questions[idx]
                gold_ans = gold_answers[idx]
                original_solution = gold_raw_answers[idx]   # ✅ 直接使用原始 solution

                # 如果压根没有抽到标准答案，就直接标记失败，不再尝试重写
                if gold_ans is None:
                    rewrite_success[idx] = False
                    continue

                prompt = build_prompt(tokenizer, q, original_solution)
                prompts.append(prompt)
                cur_indices.append(idx)

            if not prompts:
                break

            # 调用模型批量生成
            outputs = llm.generate(prompts, sampling_params)

            new_remaining_indices = []
            for idx, out in zip(cur_indices, outputs):
                gen_text = out.outputs[0].text
                gold_ans = gold_answers[idx]
                pred_ans = extract_pred_answer_from_output(gen_text)

                trial_counts[idx] += 1
                final_rewritten[idx] = gen_text

                # 校验：boxed 里的内容是否和 gold 完全一致（字符串比较）
                if gold_ans is not None and pred_ans == gold_ans:
                    rewrite_success[idx] = True
                else:
                    # 失败就重试，超过 max_retries 就放弃
                    if trial_counts[idx] <= max_retries:
                        new_remaining_indices.append(idx)
                    else:
                        rewrite_success[idx] = False
                        num_bad += 1

            remaining_indices = new_remaining_indices

        # 把当前 batch 的结果整理成记录
        for idx in batch_indices:
            ex = data[idx]
            q = questions[idx]
            gold_raw = gold_raw_answers[idx]
            gold_ans = gold_answers[idx]
            orig_reasoning = reasonings[idx]

            rewritten_answer = final_rewritten.get(idx, None)
            success_flag = rewrite_success.get(idx, False)
            trials = trial_counts.get(idx, 0)

            # 重写成功就用重写后的答案，否则回退到原始 solution
            if success_flag and rewritten_answer is not None:
                final_answer_for_train = rewritten_answer
            else:
                final_answer_for_train = gold_raw

            record = {
                "idx": idx,
                "source": ex.get("source", ""),
                "question": q,
                "gold_answer_raw": gold_raw,
                # 注意：这里名字还是 gold_answer_num，实际上是“boxed 中抽出来的字符串”
                "gold_answer_num": gold_ans,
                # 新增：保留原始 solution
                "original_solution": gold_raw,
                # 保留：去掉 boxed 的推理部分（仅用于分析 / debug）
                "original_reasoning": orig_reasoning,
                "rewritten_answer": rewritten_answer,
                "final_answer_for_train": final_answer_for_train,
                "rewrite_success": success_flag,
                "num_rewrite_trials": trials,
            }

            rewritten_records.append(record)

            if success_flag and rewritten_answer is not None:
                success_records.append(record)

    print(
        f"Rewriting done. Total samples: {len(rewritten_records)}, "
        f"rewrite failed (fallback to original) samples: {num_bad}"
    )

    # 4. 写出全量重写结果（JSON list）
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rewritten_records, f, ensure_ascii=False, indent=2)
    print(f"Full rewritten dataset saved to: {output_path}")

    # 5. 写出“重写成功子集”（JSON list）
    with success_output_path.open("w", encoding="utf-8") as f:
        json.dump(success_records, f, ensure_ascii=False, indent=2)
    print(
        f"Rewrite-success-only dataset saved to: {success_output_path}, "
        f"total success samples: {len(success_records)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite math dataset (jsonl), focusing on LaTeX boxed answers and reasoning."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the full rewritten JSON.")
    parser.add_argument("--success-output-path", type=str, default=None, help="Path to save the success-only JSON.")
    parser.add_argument("--tp-size", type=int, default=1, help="vLLM tensor_parallel_size.")
    parser.add_argument("--batch-size", type=int, default=8, help="vLLM batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per sample.")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retry attempts for failed answers.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rewrite_dataset(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        success_output_path=args.success_output_path,
        tp_size=args.tp_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
    )
