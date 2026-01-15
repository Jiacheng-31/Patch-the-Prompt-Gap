#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# -----------------------------
# Regex
# -----------------------------

BOXED_ANY_REGEX = re.compile(r"\$?\s*\\boxed\{\s*(.*?)\s*\}\s*\$?", re.DOTALL)
LETTER_IN_TEXT_REGEX = re.compile(r"([A-Z])")

# option prefix like "(A) ..." / "A. ..." / "（A）..."
OPT_PREFIX_1 = re.compile(r"^\s*[\(（]\s*[A-Z]\s*[\)）]\s*")           # (A) / （A）
OPT_PREFIX_2 = re.compile(r"^\s*[A-Z]\s*[\.\)．:：、]\s*")             # A. / A) / A： / A、


def extract_boxed_inner(text: Optional[str]) -> Optional[str]:
    """Return inner text in \\boxed{...}; None if not found."""
    if text is None:
        return None
    m = BOXED_ANY_REGEX.search(str(text))
    if not m:
        return None
    inner = m.group(1).strip()
    return inner if inner else None


def clean_option(opt: Any) -> str:
    s = "" if opt is None else str(opt)
    s = OPT_PREFIX_1.sub("", s)
    s = OPT_PREFIX_2.sub("", s)
    return s.strip()


def format_choices(choices: List[str]) -> str:
    lines = []
    for i, c in enumerate(choices):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {clean_option(c)}")
    return "\n".join(lines)


def normalize_gold_mcq_label(label: Any) -> Optional[str]:
    """
    Normalize MCQ gold label to 'A'/'B'/...
    Compatible with:
      - "D"
      - "(D)"
      - ["A"]  (some processed variants)
    """
    if label is None:
        return None
    if isinstance(label, list):
        if len(label) == 0:
            return None
        label = label[0]
    s = str(label).strip().upper()
    m = LETTER_IN_TEXT_REGEX.search(s)
    return m.group(1) if m else None


def extract_pred_mcq_letter(
    model_output: Optional[str],
    n_choices: int,
    strict_boxed: bool,
    fallback_last_letter: bool,
) -> Optional[str]:
    """
    MCQ prediction extraction.
    - strict_boxed: must have \\boxed{...}
    - fallback_last_letter: if strict fails, try last A-Z letter in whole output
    """
    text = "" if model_output is None else str(model_output)

    def in_range(ch: str) -> bool:
        if n_choices <= 0:
            return False
        max_letter = chr(ord("A") + n_choices - 1)
        return "A" <= ch <= max_letter

    # 1) boxed
    inner = extract_boxed_inner(text)
    if inner is not None:
        letters = LETTER_IN_TEXT_REGEX.findall(inner.upper())
        if letters:
            cand = letters[-1]
            if in_range(cand):
                return cand

    if strict_boxed and not fallback_last_letter:
        return None

    # 2) fallback: last letter anywhere
    letters2 = LETTER_IN_TEXT_REGEX.findall(text.upper())
    if not letters2:
        return None
    cand2 = letters2[-1]
    return cand2 if in_range(cand2) else None


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


_NUM_REGEX = re.compile(r"[\-+]?\d+(?:\.\d+)?")

def try_parse_last_number(s: str) -> Optional[float]:
    nums = _NUM_REGEX.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def cloze_match(gold: Any, pred_text: Optional[str], strict_boxed: bool) -> bool:
    """
    Cloze scoring:
    - if strict_boxed: require boxed
    - gold can be string or list of strings
    - numeric compare if both parse as number else case-insensitive exact match after normalize
    """
    if pred_text is None:
        return False

    if strict_boxed:
        pred = extract_boxed_inner(pred_text)
        if pred is None:
            return False
    else:
        pred = extract_boxed_inner(pred_text) or pred_text

    pred_norm = normalize_text(pred)

    if isinstance(gold, list):
        return any(cloze_match(g, pred_norm, strict_boxed=False) for g in gold)

    gold_norm = normalize_text(gold)
    if not gold_norm:
        return False

    gnum = try_parse_last_number(gold_norm)
    pnum = try_parse_last_number(pred_norm)
    if gnum is not None and pnum is not None:
        return abs(gnum - pnum) <= 1e-6

    return gold_norm.casefold() == pred_norm.casefold()


# -----------------------------
# Prompts (NOTE: double braces for literal braces in .format templates)
# -----------------------------

MCQ_PROMPT_TEMPLATE = (
    "You are an expert problem solver.\n"
    "Read the following multiple-choice question and solve it step by step.\n"
    "Choose the best option.\n"
    "At the end, on a separate line, output ONLY the final option letter "
    "in LaTeX boxed format, exactly like:\n"
    "$\\boxed{{A}}$\n"
    "Use dollar signs and \\boxed{{}} exactly as shown.\n"
    "{passage_block}"
    "Question: {question}\n"
    "Options:\n{options}\n"
)

CLOZE_PROMPT_TEMPLATE = (
    "You are an expert problem solver.\n"
    "Solve the following question.\n"
    "At the end, on a separate line, output ONLY the final answer "
    "in LaTeX boxed format, exactly like:\n"
    "$\\boxed{{your_answer}}$\n"
    "Use dollar signs and \\boxed{{}} exactly as shown.\n"
    "{passage_block}"
    "Question: {question}\n"
)


def build_prompts(
    tokenizer,
    passages: List[Optional[str]],
    questions: List[str],
    choices_list: List[List[str]],
) -> List[str]:
    prompts: List[str] = []
    for p, q, choices in zip(passages, questions, choices_list):
        passage_block = ""
        if p is not None and str(p).strip():
            passage_block = f"Passage:\n{p}\n\n"

        if choices and len(choices) > 0:
            user_content = MCQ_PROMPT_TEMPLATE.format(
                passage_block=passage_block,
                question=q,
                options=format_choices(choices),
            )
        else:
            user_content = CLOZE_PROMPT_TEMPLATE.format(
                passage_block=passage_block,
                question=q,
            )

        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


# -----------------------------
# Data IO (robust JSONL)
# -----------------------------

def load_json_or_jsonl(file_path: Path, skip_bad_lines: bool) -> List[Dict[str, Any]]:
    """
    Robust JSON/JSONL loader.
    - Uses encoding='utf-8-sig' to skip UTF-8 BOM automatically.
    - If skip_bad_lines: warn and continue; else raise with file+line info.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        data: List[Dict[str, Any]] = []
        bad = 0
        with file_path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # if BOM somehow appears in middle (rare concatenation issue)
                line = line.lstrip("\ufeff")

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    bad += 1
                    preview = line[:200].replace("\n", "\\n")
                    msg = f"[WARN] JSONL parse error in {file_path.name} line {lineno}: {e} | preview={preview!r}"
                    if skip_bad_lines:
                        print(msg)
                        continue
                    raise ValueError(msg) from e

                if not isinstance(obj, dict):
                    bad += 1
                    msg = f"[WARN] {file_path.name} line {lineno} is not a JSON object(dict). got={type(obj)}"
                    if skip_bad_lines:
                        print(msg)
                        continue
                    raise ValueError(msg)

                data.append(obj)

        if bad > 0:
            print(f"[INFO] {file_path.name}: skipped bad lines = {bad}")
        return data

    # .json list
    text = file_path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not text:
        return []
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"{file_path} is not a JSON list. Top-level type={type(obj)}")


# -----------------------------
# Eval
# -----------------------------

@dataclass
class EvalStats:
    file: str
    n_total: int
    n_correct: int
    n_no_pred: int
    acc: float


DEFAULT_MATH_LIKE = {"aqua-rat.jsonl", "sat-math.jsonl"}


def discover_files(data_dir: Path, include_math: bool, files_arg: Optional[List[str]]) -> List[Path]:
    if files_arg and len(files_arg) > 0:
        files = [data_dir / f for f in files_arg]
        return [p for p in files if p.is_file()]

    files = sorted(data_dir.glob("*.jsonl"))
    if not include_math:
        files = [p for p in files if p.name not in DEFAULT_MATH_LIKE]
    return files


def evaluate_single_file(
    llm: LLM,
    tokenizer,
    file_path: Path,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    strict_boxed: bool,
    fallback_last_letter: bool,
    skip_bad_lines: bool,
    output_dir: Optional[Path],
) -> EvalStats:
    data = load_json_or_jsonl(file_path, skip_bad_lines=skip_bad_lines)

    passages: List[Optional[str]] = []
    questions: List[str] = []
    choices_list: List[List[str]] = []
    is_mcq: List[bool] = []
    gold_mcq: List[Optional[str]] = []
    gold_cloze: List[Any] = []
    qids: List[Any] = []

    for ex in data:
        q = ex.get("question", None)
        if q is None:
            continue

        p = ex.get("passage", None)
        options = ex.get("choices") or ex.get("options") or []
        mcq_flag = bool(options and len(options) > 0)

        if mcq_flag:
            gold = normalize_gold_mcq_label(ex.get("label", None))  # MCQ gold in label
            gold_mcq.append(gold)
            gold_cloze.append(None)
        else:
            gold = ex.get("answer", None)  # cloze gold in answer
            gold_mcq.append(None)
            gold_cloze.append(gold)

        passages.append(p)
        questions.append(str(q))
        choices_list.append([str(x) for x in options] if options else [])
        is_mcq.append(mcq_flag)
        qids.append(ex.get("question_id", None))

    n_total = len(questions)
    print(f"\nLoaded {file_path.name}, total samples = {n_total}")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    n_correct = 0
    n_no_pred = 0
    all_records: List[Dict[str, Any]] = []

    for start in tqdm(range(0, n_total, batch_size), desc=f"Eval {file_path.name}"):
        end = min(start + batch_size, n_total)

        batch_prompts = build_prompts(
            tokenizer,
            passages[start:end],
            questions[start:end],
            choices_list[start:end],
        )
        outputs = llm.generate(batch_prompts, sampling_params)

        for i, out in enumerate(outputs):
            idx = start + i
            pred_text = out.outputs[0].text if out.outputs else ""

            mcq_flag = is_mcq[idx]
            choices = choices_list[idx]

            correct = False
            pred_extracted: Any = None

            if mcq_flag:
                gold = gold_mcq[idx]
                pred_letter = extract_pred_mcq_letter(
                    pred_text,
                    n_choices=len(choices),
                    strict_boxed=strict_boxed,
                    fallback_last_letter=fallback_last_letter,
                )
                pred_extracted = pred_letter

                if pred_letter is None:
                    n_no_pred += 1
                correct = (gold is not None) and (pred_letter is not None) and (gold == pred_letter)
            else:
                gold = gold_cloze[idx]
                ok = cloze_match(gold, pred_text, strict_boxed=strict_boxed)
                pred_extracted = extract_boxed_inner(pred_text) if strict_boxed else (extract_boxed_inner(pred_text) or pred_text)
                if strict_boxed and pred_extracted is None:
                    n_no_pred += 1
                correct = ok

            if correct:
                n_correct += 1

            if output_dir is not None:
                rec = {
                    "index": idx,
                    "question_id": qids[idx],
                    "is_mcq": mcq_flag,
                    "passage": passages[idx],
                    "question": questions[idx],
                    "options": choices,
                    "gold_label": gold_mcq[idx],
                    "gold_answer": gold_cloze[idx],
                    "pred": pred_extracted,
                    "model_output": pred_text,
                    "correct": bool(correct),
                    "has_boxed": bool(extract_boxed_inner(pred_text) is not None),
                }
                all_records.append(rec)

    acc = n_correct / n_total if n_total > 0 else 0.0
    print(f"Accuracy on {file_path.name}: {n_correct} / {n_total} = {acc:.4%}")
    print(f"Samples with no extracted answer (boxed missing/invalid): {n_no_pred}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{file_path.stem}.pred.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved per-sample results to: {out_path}")

    return EvalStats(file=file_path.name, n_total=n_total, n_correct=n_correct, n_no_pred=n_no_pred, acc=acc)


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AGIEval v1_1 evaluation (vLLM). MCQ gold uses `label`, cloze gold uses `answer`."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True, help="Path to AGIEval data/v1_1 directory")
    parser.add_argument("--files", type=str, nargs="*", default=None,
                        help="Optional explicit file list. If omitted, auto-scan *.jsonl.")
    parser.add_argument("--include-math", action="store_true",
                        help="Include math-like tasks (default excludes aqua-rat.jsonl and sat-math.jsonl).")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)

    # default strict boxed ON; use --no-strict-boxed to disable
    parser.add_argument("--no-strict-boxed", dest="strict_boxed", action="store_false")
    parser.set_defaults(strict_boxed=True)

    parser.add_argument("--fallback-last-letter", action="store_true",
                        help="If strict boxed fails on MCQ, fallback to last A-Z letter in output (reduces no_pred).")

    # default skip bad lines ON (prevents mid-run crash)
    parser.add_argument("--no-skip-bad-lines", dest="skip_bad_lines", action="store_false")
    parser.set_defaults(skip_bad_lines=True)

    parser.add_argument("--output-dir", type=str, default=None, help="If set, save per-sample jsonl predictions here.")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"--data-dir not a directory: {data_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    files = discover_files(data_dir, include_math=args.include_math, files_arg=args.files)
    if not files:
        raise ValueError("No data files found. Provide --files or check --data-dir.")

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(
        f"Initializing vLLM LLM from {args.model_path} "
        f"with tensor_parallel_size={args.tp_size} ..."
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
    )

    print("\nFiles to evaluate:")
    for p in files:
        print(" -", p.name)

    all_stats: List[EvalStats] = []
    for file_path in files:
        stats = evaluate_single_file(
            llm=llm,
            tokenizer=tokenizer,
            file_path=file_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            strict_boxed=args.strict_boxed,
            fallback_last_letter=args.fallback_last_letter,
            skip_bad_lines=args.skip_bad_lines,
            output_dir=output_dir,
        )
        all_stats.append(stats)

    print("\n===== Summary across files =====")
    total_n = sum(s.n_total for s in all_stats)
    total_correct = sum(s.n_correct for s in all_stats)
    total_no_pred = sum(s.n_no_pred for s in all_stats)
    overall_acc = total_correct / total_n if total_n > 0 else 0.0

    for s in all_stats:
        print(f"{s.file:28s}  {s.n_correct:5d}/{s.n_total:5d}  acc={s.acc:.4%}  no_pred={s.n_no_pred}")

    print(f"\nOverall accuracy: {total_correct} / {total_n} = {overall_acc:.4%}")
    print(f"Total no_pred (boxed missing/invalid): {total_no_pred}")


if __name__ == "__main__":
    main()
