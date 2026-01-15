import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ====== 从 \boxed{...} 抽取任意内容 ======
BOXED_ANY_REGEX = re.compile(r"\$?\s*\\boxed\{\s*(.*?)\s*\}\s*\$?", re.DOTALL)

# ====== 抽取单选字母（A-Z） ======
LETTER_IN_TEXT_REGEX = re.compile(r"([A-Z])")


def extract_boxed_inner(text: str | None) -> str | None:
    """严格：只在 boxed 里找；找不到 boxed 返回 None。"""
    if text is None:
        return None
    m = BOXED_ANY_REGEX.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    return inner if inner else None


def extract_pred_letter_strict_nofail(model_output: str | None, choices: list[str] | None) -> str | None:
    """
    严格但不报错：
    - 必须有 \\boxed{...}，否则返回 None
    - boxed 内必须能解析出一个选项字母（A-Z）
    - 若 choices 给了，则字母必须在范围 A..(A+len(choices)-1)
    - 任意一步失败都返回 None
    """
    inner = extract_boxed_inner(model_output)
    if inner is None:
        return None

    inner_up = inner.upper()
    letters = LETTER_IN_TEXT_REGEX.findall(inner_up)
    if not letters:
        return None

    pred = letters[-1]  # 取最后一个字母，避免 'Answer' 的 A 干扰

    if choices is not None and len(choices) > 0:
        max_letter = chr(ord("A") + len(choices) - 1)
        if pred < "A" or pred > max_letter:
            return None

    return pred


def normalize_gold_answer(ans: Any) -> str | None:
    """gold answer 期望是 'A'/'B'/...；解析失败返回 None。"""
    if ans is None:
        return None
    s = str(ans).strip().upper()
    if not s:
        return None
    m = LETTER_IN_TEXT_REGEX.search(s)
    return m.group(1) if m else None


def format_choices(choices: list[str]) -> str:
    lines = []
    for i, c in enumerate(choices):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {c}")
    return "\n".join(lines)


PROMPT_TEMPLATE = (
    "You are an expert problem solver.\n"
    "Read the following multiple-choice question and solve it step by step.\n"
    "Choose the best option.\n"
    "At the end, on a separate line, output ONLY the final option letter "
    "in LaTeX boxed format, exactly like:\n"
    "$\\boxed{{A}}$\n"
    "Use dollar signs and \\boxed{{}} exactly as shown.\n"
    "Question: {question}\n"
    "Options:\n{options}"
)


def build_prompts(tokenizer, questions: list[str], choices_list: list[list[str]]) -> list[str]:
    prompts: list[str] = []
    for q, choices in zip(questions, choices_list):
        if q is None:
            q = ""
        if choices is None:
            choices = []

        safe_q = q.replace("{", "{{").replace("}", "}}")
        safe_opts = format_choices(choices).replace("{", "{{").replace("}", "}}")
        user_content = PROMPT_TEMPLATE.format(question=safe_q, options=safe_opts)

        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def load_json_or_jsonl(file_path: Path) -> list[dict]:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if file_path.suffix.lower() == ".jsonl":
        data = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"{file_path} is not a JSON list. Top-level type={type(obj)}")


def evaluate_single_file(
    llm: LLM,
    tokenizer,
    file_path: Path,
    batch_size: int,
    max_new_tokens: int,
    output_dir: Path | None = None,
):
    data = load_json_or_jsonl(file_path)

    questions: list[str] = []
    choices_list: list[list[str]] = []
    gold_letters: list[str | None] = []
    qids: list[Any] = []

    for ex in data:
        q = ex.get("question")
        choices = ex.get("choices") or ex.get("options") or []
        ans = ex.get("answer")
        qid = ex.get("question_id", None)

        if q is None:
            continue

        questions.append(str(q))
        choices_list.append([str(x) for x in choices])
        gold_letters.append(normalize_gold_answer(ans))
        qids.append(qid)

    n_total = len(questions)
    print(f"\nLoaded {file_path.name}, total samples = {n_total}")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    n_correct = 0
    n_no_pred = 0
    all_records = []

    for start in tqdm(range(0, n_total, batch_size), desc=f"Eval {file_path.name}"):
        end = min(start + batch_size, n_total)
        batch_q = questions[start:end]
        batch_choices = choices_list[start:end]
        batch_gold = gold_letters[start:end]
        batch_qids = qids[start:end]

        batch_prompts = build_prompts(tokenizer, batch_q, batch_choices)
        outputs = llm.generate(batch_prompts, sampling_params)

        for i, out in enumerate(outputs):
            idx = start + i
            q = batch_q[i]
            choices = batch_choices[i]
            gold = batch_gold[i]
            qid = batch_qids[i]

            pred_text = out.outputs[0].text
            pred_letter = extract_pred_letter_strict_nofail(pred_text, choices)

            if pred_letter is None:
                n_no_pred += 1

            correct = (gold is not None) and (pred_letter is not None) and (gold == pred_letter)
            if correct:
                n_correct += 1

            if output_dir is not None:
                rec = {
                    "index": idx,
                    "question_id": qid,
                    "question": q,
                    "choices": choices,
                    "gold_answer": gold,
                    "pred_answer": pred_letter,
                    "model_output": pred_text,
                    "correct": bool(correct),
                    "has_boxed": bool(extract_boxed_inner(pred_text) is not None),
                }
                all_records.append(rec)

    acc = n_correct / n_total if n_total > 0 else 0.0
    print(f"Accuracy on {file_path.name}: {n_correct} / {n_total} = {acc:.4%}")
    print(f"Samples with no extracted boxed answer on {file_path.name}: {n_no_pred}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{file_path.stem}.pred.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Detailed per-sample results saved to: {out_path}")

    return {
        "file": file_path.name,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_no_pred": n_no_pred,
        "acc": acc,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict evaluation for MMLU/MMLU-Pro: must answer with \\boxed{letter}; missing/invalid counted as false."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

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

    if all_stats:
        print("\n===== Summary across all files =====")
        total_n = sum(s["n_total"] for s in all_stats)
        total_correct = sum(s["n_correct"] for s in all_stats)
        total_no_pred = sum(s["n_no_pred"] for s in all_stats)
        overall_acc = total_correct / total_n if total_n > 0 else 0.0
        print(f"Overall accuracy: {total_correct} / {total_n} = {overall_acc:.4%}")
        print(f"Total samples with no extracted boxed answer: {total_no_pred}")
    else:
        print("No valid files evaluated.")


if __name__ == "__main__":
    main()
