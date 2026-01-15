import argparse
import json
import math
import re
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ====== 正则：提取 \boxed{...} 里的“数字答案”（与之前代码一致） ======
BOXED_ANSWER_REGEX = re.compile(
    r"\$?\s*\\boxed\{\s*([\-+]?\d+(?:\.\d+)?)\s*\}\s*\$?"
)

# ====== 新增：提取 \boxed{...} 里的任意内容（用于非数值答案） ======
BOXED_ANY_REGEX = re.compile(
    r"\$?\s*\\boxed\{\s*(.*?)\s*\}\s*\$?", re.DOTALL
)


def extract_gold_answer(text: str | None) -> str | None:
    """
    从 gold 文本中抽取最终的数字答案。
    优先匹配 '\\boxed{number}'，如果没有，则退化为取最后一个数字。
    （逻辑沿用你之前的 AIME/AMC 脚本）
    """
    if text is None:
        return None

    m = BOXED_ANSWER_REGEX.search(text)
    if m:
        return m.group(1).strip()

    nums = re.findall(r"[\-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()

    return None


def extract_pred_answer_from_output(text: str | None) -> str | None:
    """
    从模型输出中抽取预测的“数值”答案。
    （保持与之前 AIME/AMC 代码一致）
    """
    if text is None:
        return None

    m = BOXED_ANSWER_REGEX.search(text)
    if m:
        return m.group(1).strip()

    nums = re.findall(r"[\-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()

    return None


def normalize_number_str(s: str | None) -> str | None:
    """
    把数字字符串标准化一下，避免 18 vs 18.0 这种差异。
    （逻辑与之前 AIME/AMC 脚本一致）

    修改点：如果解析出来是 inf / -inf / NaN（非有限数），
    直接原样返回，避免 round(inf) / round(NaN) 抛 OverflowError。
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None

    if s.startswith("+"):
        s = s[1:]

    try:
        v = float(s)
    except ValueError:
        # 无法解析为 float，就原样返回（比如分数 "3/5"）
        return s

    # 如果是 inf、-inf 或 NaN，直接原样返回，避免 round 抛错
    if not math.isfinite(v):
        return s

    # 整数：统一成纯整数字符串
    if math.isclose(v, round(v)):
        return str(int(round(v)))
    else:
        # 否则保留到 6 位小数，然后去掉多余的 0 和小数点
        s2 = f"{v:.6f}".rstrip("0").rstrip(".")
        return s2


# ====== 新增：判定 gold Answer 是不是“非数值类型”（字母 / 比例） ======
def is_textual_gold_answer(ans: str | None) -> bool:
    """
    True: 当作非纯数值答案（比如 'B, C'、'3:1'）
    False: 当作纯数值题（比如 '42', '\\boxed{15}', '3.14'）
    """
    if ans is None:
        return False
    s = str(ans).strip()
    if not s:
        return False

    # 1) 含有字母：通常就是选项题（A/B/C/D）
    if re.search(r"[A-Za-z]", s):
        return True

    # 2) 纯比例形式：3:1 这种（两边都是数字）
    if re.fullmatch(r"([\-+]?\d+(?:\.\d+)?)\s*:\s*([\-+]?\d+(?:\.\d+)?)", s):
        return True

    return False


# ====== 新增：归一化“非数值”答案（字母/比例等） ======
def normalize_text_answer(s: str | None) -> str | None:
    """
    归一化非数值答案，使得：
    - 'B, C'  /  'C B'  /  '(B,C)' --> 'BC'
    - '3:1'   /  '3 : 1'          --> '3:1'
    - 其它字符串：统一大写、压缩空格、去掉末尾标点
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    # 去掉最外层的 $...$ 或 \( ... \) / \[ ... \]
    if s.startswith("$") and s.endswith("$") and len(s) > 2:
        s = s[1:-1].strip()
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()

    # 1) 比例形式 3:1
    m_ratio = re.fullmatch(r"([\-+]?\d+(?:\.\d+)?)\s*:\s*([\-+]?\d+(?:\.\d+)?)", s)
    if m_ratio:
        left = normalize_number_str(m_ratio.group(1))
        right = normalize_number_str(m_ratio.group(2))
        if left is None or right is None:
            return f"{m_ratio.group(1).strip()}:{m_ratio.group(2).strip()}"
        return f"{left}:{right}"

    # 2) 多选字母：B, C  /  A and D  /  (A,B,C) ...
    letters = re.findall(r"[A-Za-z]", s)
    # 只包含字母（没有数字）才按选项处理，避免把别的东西误伤
    if letters and not re.search(r"\d", s):
        letters = sorted(set(ch.upper() for ch in letters))
        return "".join(letters)

    # 3) 其它情况：统一大写 + 压缩空格 + 去末尾标点
    s_up = s.upper()
    s_up = re.sub(r"\s+", " ", s_up)
    s_up = s_up.strip(" .;,")
    return s_up or None


# ====== 新增：从模型输出中抽取“文本答案”（不限数字） ======
def extract_text_answer_from_output(text: str | None) -> str | None:
    """
    优先从 \\boxed{...} 中取出任意内容，
    若没有 boxed，则退化为取最后一行非空文本。
    """
    if text is None:
        return None

    m = BOXED_ANY_REGEX.search(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner

    # 没有 boxed，就取最后一行非空文本
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]


# ====== prompt 模板（保持与之前完全一致） ======
PROMPT_TEMPLATE = (
    "You are an expert in math word problems.\n"
    "Read the following problem and solve it step by step.\n"
    "At the end, on a separate line, output ONLY the final answer "
    "in LaTeX boxed format, exactly like:\n"
    "$\\boxed{{56}}$\n"
    "Use dollar signs and \\boxed{{}} exactly as shown.\n"
    "Problem: {question}"
)


def build_prompts(tokenizer, questions: list[str]) -> list[str]:
    """
    与之前 orca_math / AIME / AMC 代码一致的 prompt 构造逻辑：
    - 对题目里的 { } 进行转义
    - 使用 tokenizer.apply_chat_template 构造对话式 prompt
    """
    prompts: list[str] = []

    for q in questions:
        if q is None:
            q = ""
        # 转义题目里的大括号，避免 format 把它们当占位符
        safe_q = q.replace("{", "{{").replace("}", "}}")

        user_content = PROMPT_TEMPLATE.format(question=safe_q)

        messages = [
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    return prompts


def evaluate_single_file(
    llm: LLM,
    tokenizer,
    file_path: Path,
    batch_size: int,
    max_new_tokens: int,
    output_dir: Path | None = None,
):
    """
    评测单个 JSONL 文件：
    - 假定字段为 Problem / Answer
    - Answer 可能是字符串（含字母、比例）也可能是数字
    - 数值题：使用原来的数值答案逻辑
    - 非数值题：使用文本归一化逻辑
    """
    data = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)

    questions: list[str] = []
    gold_answers_raw: list[str | None] = []

    for ex in data:
        q = ex.get("Problem")
        a = ex.get("Answer")
        if q is None:
            continue
        questions.append(q)
        # Answer 可能是数字也可能是字符串，这里统一 str 化，保持兼容性
        if a is None:
            gold_answers_raw.append(None)
        else:
            gold_answers_raw.append(str(a))

    n_total = len(questions)
    print(f"\nLoaded test set from {file_path}, total samples = {n_total}")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    n_correct = 0
    n_no_pred = 0  # 现在含义：没有成功抽到可用答案（数值 or 文本）
    all_records = []

    for start in tqdm(range(0, n_total, batch_size), desc=f"Eval {file_path.name}"):
        end = min(start + batch_size, n_total)
        batch_questions = questions[start:end]
        batch_gold_raw = gold_answers_raw[start:end]

        batch_prompts = build_prompts(tokenizer, batch_questions)
        outputs = llm.generate(batch_prompts, sampling_params)

        for i, out in enumerate(outputs):
            q_idx = start + i
            q = batch_questions[i]
            gold_raw = batch_gold_raw[i]

            pred_text = out.outputs[0].text

            # 判断这一条是不是“非数值答案”
            textual = is_textual_gold_answer(gold_raw)

            gold_num = pred_num = None
            gold_text_norm = pred_text_norm = pred_text_extracted = None

            if not textual:
                # ===== 数值题路径：保持与原来完全一致（只是 normalize 更健壮） =====
                gold_num = normalize_number_str(extract_gold_answer(gold_raw))
                pred_num = normalize_number_str(
                    extract_pred_answer_from_output(pred_text)
                )

                correct = (
                    (gold_num is not None)
                    and (pred_num is not None)
                    and (gold_num == pred_num)
                )
                if pred_num is None:
                    n_no_pred += 1
            else:
                # ===== 非数值题路径：比如 Answer = 'B, C' 或 '3:1' =====
                gold_text_norm = normalize_text_answer(gold_raw)
                pred_text_extracted = extract_text_answer_from_output(pred_text)
                pred_text_norm = normalize_text_answer(pred_text_extracted)

                correct = (
                    (gold_text_norm is not None)
                    and (pred_text_norm is not None)
                    and (gold_text_norm == pred_text_norm)
                )
                if pred_text_norm is None:
                    n_no_pred += 1

            if correct:
                n_correct += 1

            if output_dir is not None:
                rec = {
                    "index": q_idx,
                    "problem": q,
                    "gold_answer_raw": gold_raw,
                    "model_output": pred_text,
                    "correct": bool(correct),
                    "is_textual": bool(textual),
                }
                if not textual:
                    rec["gold_answer_num"] = gold_num
                    rec["pred_answer_num"] = pred_num
                else:
                    rec["gold_answer_text_norm"] = gold_text_norm
                    rec["pred_answer_text_raw_extracted"] = pred_text_extracted
                    rec["pred_answer_text_norm"] = pred_text_norm

                all_records.append(rec)

    acc = n_correct / n_total if n_total > 0 else 0.0
    print(
        f"Accuracy on {file_path.name}: {n_correct} / {n_total} = {acc:.4%}"
    )
    print(
        f"Samples with no extracted answer (numeric or textual) on {file_path.name}: {n_no_pred}"
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{file_path.stem}.pred.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Detailed per-sample results saved to: {out_path}")

    # 返回统计信息，方便主程序汇总
    return {
        "file": file_path.name,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_no_pred": n_no_pred,
        "acc": acc,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple math JSONL test sets (Problem/Answer) with vLLM. "
            "Numeric path matches original AIME/AMC scripts; also supports non-numeric answers like 'B, C' and '3:1'."
        )
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing the JSONL test files.",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of JSONL filenames to evaluate, relative to data-dir. "
            "Each file must contain fields: Problem / Answer."
        ),
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, write per-sample prediction jsonl to this directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = None

    # ====== 只加载一次 tokenizer 和 vLLM 模型 ======
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(
        f"Initializing vLLM LLM from {args.model_path} "
        f"with tensor_parallel_size={args.tp_size} ..."
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
    )

    # ====== 依次评测多个文件 ======
    all_stats = []
    for fname in args.files:
        file_path = data_dir / fname
        if not file_path.is_file():
            print(f"[WARN] File not found: {file_path}, skip.")
            continue

        stats = evaluate_single_file(
            llm=llm,
            tokenizer=tokenizer,
            file_path=file_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            output_dir=output_dir,
        )
        all_stats.append(stats)

    # ====== 汇总打印一下整体信息 ======
    if all_stats:
        print("\n===== Summary across all files =====")
        total_n = sum(s["n_total"] for s in all_stats)
        total_correct = sum(s["n_correct"] for s in all_stats)
        total_no_pred = sum(s["n_no_pred"] for s in all_stats)
        overall_acc = total_correct / total_n if total_n > 0 else 0.0
        print(f"Overall accuracy: {total_correct} / {total_n} = {overall_acc:.4%}")
        print(f"Total samples with no extracted answer: {total_no_pred}")
    else:
        print("No valid files evaluated.")


if __name__ == "__main__":
    main()
