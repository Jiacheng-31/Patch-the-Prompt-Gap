# -*- coding: utf-8 -*-
import re
import math
from threading import Lock
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 1) boxed / 数字 =====================
BOX_PATTERN = re.compile(r"\$?\s*\\boxed\{(.*?)\}\s*\$?", re.DOTALL)

def extract_boxed_or_last_number(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text)

    m = BOX_PATTERN.search(text)
    if m:
        return m.group(1).strip()

    nums = re.findall(r"[\-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()
    return None

def normalize(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

# ===================== 2) scoring model 单例 =====================
SCORING_MODEL_PATH = "/workspace/models/Mistral-7B-Instruct-v0.3"
MAX_SEQ_LEN = 4096

_scoring_model = None
_scoring_tokenizer = None
_load_lock = Lock()

def _safe_load_model(model_path: str, device: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=False)

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map=None, **base_kwargs
        )
    except TypeError:
        pass

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=None, **base_kwargs
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, **base_kwargs
        )

def _get_scoring_model() -> Tuple[Any, Any]:
    global _scoring_model, _scoring_tokenizer
    with _load_lock:
        if _scoring_model is None or _scoring_tokenizer is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[rewrite_reward] loading scoring model from: {SCORING_MODEL_PATH} on {device}")

            _scoring_tokenizer = AutoTokenizer.from_pretrained(
                SCORING_MODEL_PATH, trust_remote_code=True
            )
            if getattr(_scoring_tokenizer, "pad_token_id", None) is None:
                _scoring_tokenizer.pad_token = _scoring_tokenizer.eos_token

            _scoring_model = _safe_load_model(SCORING_MODEL_PATH, device).to(device)
            _scoring_model.eval()

            try:
                if _scoring_model.get_input_embeddings().weight.shape[0] != len(_scoring_tokenizer):
                    _scoring_model.resize_token_embeddings(len(_scoring_tokenizer))
            except Exception:
                pass
            try:
                _scoring_model.tie_weights()
            except Exception:
                pass
            try:
                _scoring_model.config.pad_token_id = _scoring_tokenizer.pad_token_id
            except Exception:
                pass

            print(f"[rewrite_reward] scoring model device: {next(_scoring_model.parameters()).device}")

    return _scoring_model, _scoring_tokenizer

# ===================== 3) v_rea: 轻量 judge（YES/NO logprob） =====================
JUDGE_PROMPT = (
    "You are a strict math solution verifier.\n"
    "Given a problem, a reference final answer, and a candidate solution, "
    "decide whether the candidate reasoning is logically valid and consistent with the final answer.\n"
    "Reply with ONLY one token: YES or NO.\n\n"
    "Problem:\n{question}\n\n"
    "Reference final answer:\n{gold}\n\n"
    "Candidate solution:\n{solution}\n\n"
    "Decision:"
)

def _judge_reasoning_yesno(question: str, gold_boxed: str, solution: str) -> int:
    """
    返回 1/0。只在 v_ans=1 时调用。
    用模型对下一个 token 预测 YES/NO 的相对概率（不生成长文本）。
    """
    if not question or not gold_boxed or not solution:
        return 0

    model, tokenizer = _get_scoring_model()
    device = next(model.parameters()).device

    prompt = JUDGE_PROMPT.format(question=question, gold=gold_boxed, solution=solution)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, -1, :]  # next-token logits

    # 取 "YES" 和 "NO" 的 token（有的 tokenizer 可能是带空格的）
    cand_tokens = ["YES", " NO", "NO", " YES"]
    token_ids = []
    for tok in cand_tokens:
        tid = tokenizer.encode(tok, add_special_tokens=False)
        if len(tid) == 1:
            token_ids.append((tok.strip(), tid[0]))
    # 去重
    token_map = {}
    for k, v in token_ids:
        token_map[k] = v
    if "YES" not in token_map or "NO" not in token_map:
        # 兜底：无法稳定取到单 token 时，直接不给 reasoning 分
        return 0

    yes_id = token_map["YES"]
    no_id = token_map["NO"]

    probs = F.softmax(logits, dim=-1)
    p_yes = float(probs[0, yes_id].detach().cpu())
    p_no = float(probs[0, no_id].detach().cpu())

    return 1 if p_yes >= p_no else 0

# ===================== 4) dist: QA 条件下的 length-norm NLL =====================
QA_PREFIX = "Problem:\n{question}\n\nAnswer:\n"

def _length_norm_nll_under_pi0(question: str, solution: str) -> float:
    """
    计算 ell_dist = - (1/|y|) sum log p(y_t | x, y_<t)
    这里的 x 只包含 question（QA-style），避免 rewriting prompt 的条件偏置。
    """
    if not question or not solution:
        return float("inf")

    model, tokenizer = _get_scoring_model()
    device = next(model.parameters()).device

    prefix = QA_PREFIX.format(question=question)
    full = prefix + solution

    with torch.no_grad():
        prefix_ids = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        full_ids = tokenizer(full, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)

        input_ids = full_ids["input_ids"].to(device)
        attn = full_ids["attention_mask"].to(device)

        prefix_len = prefix_ids["input_ids"].shape[1]
        seq_len = input_ids.shape[1]
        prefix_len = min(prefix_len, seq_len)
        if prefix_len >= seq_len:
            return float("inf")

        labels = input_ids.clone()
        labels[:, :prefix_len] = -100

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        # 这是 mean over unmasked tokens 的 CE loss，即 per-token NLL
        ell = float(out.loss.detach().cpu())
    return max(ell, 0.0)

# ===================== 5) diversity: 用 last hidden mean pooling 当 embedding =====================
def _embed_text(text: str) -> torch.Tensor:
    model, tokenizer = _get_scoring_model()
    device = next(model.parameters()).device

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        h = out.hidden_states[-1]  # [1, T, H]

    mask = attn.unsqueeze(-1).to(h.dtype)  # [1, T, 1]
    summed = (h * mask).sum(dim=1)         # [1, H]
    denom = mask.sum(dim=1).clamp(min=1.0) # [1, 1]
    mean = summed / denom                  # [1, H]
    vec = mean[0].detach().cpu()           # CPU 上算相似度就行
    vec = vec / (vec.norm(p=2) + 1e-12)
    return vec

def _cos_dist(u: torch.Tensor, v: torch.Tensor) -> float:
    # u,v 已归一化
    cos = float((u * v).sum().item())
    return (1.0 - cos) / 2.0  # [0,1]

def _set_diversity(embs: List[torch.Tensor]) -> float:
    m = len(embs)
    if m < 2:
        return 0.0
    s = 0.0
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            s += _cos_dist(embs[i], embs[j])
            cnt += 1
    return s / max(cnt, 1)

# ===================== 6) 组内 gated reward =====================
def compute_group_rewards_for_one_prompt(
    question: str,
    solutions: List[str],
    ground_truth: str,
    lambda_dist: float = 0.2,
    lambda_div: float = 0.2,
    eps: float = 1e-6,
    enable_reasoning_judge: bool = True,
) -> List[float]:
    """
    输入同一个 x 的 K 个候选 solutions，输出 K 个 reward。
    """
    K = len(solutions)
    rewards = [0.0] * K

    # ---- v_ans：最终答案
    gold = normalize(extract_boxed_or_last_number(ground_truth))

    v_ans = []
    for sol in solutions:
        pred = normalize(extract_boxed_or_last_number(sol))
        v_ans.append(1 if (pred is not None and gold is not None and pred == gold) else 0)

    # ---- v_rea：只对 v_ans=1 的样本判
    v_rea = [0] * K
    if enable_reasoning_judge and question and gold is not None:
        for i, sol in enumerate(solutions):
            if v_ans[i] == 1:
                v_rea[i] = _judge_reasoning_yesno(question=question, gold_boxed=gold, solution=sol)
    else:
        # 关掉 judge 时：把 v_rea 当 1（只要答案对就算可行）
        for i in range(K):
            v_rea[i] = 1 if v_ans[i] == 1 else 0

    r_task = [v_ans[i] * v_rea[i] for i in range(K)]  # 0/1

    # ---- dist：只对可行集合 S_x^+ 计算组内标准化
    feasible_idx = [i for i in range(K) if r_task[i] == 1]
    ell = [float("inf")] * K
    r_dist = [0.0] * K

    if len(feasible_idx) > 0:
        for i in feasible_idx:
            ell[i] = _length_norm_nll_under_pi0(question=question, solution=solutions[i])

        feas_ells = [ell[i] for i in feasible_idx if math.isfinite(ell[i])]
        if len(feas_ells) >= 2:
            mu = sum(feas_ells) / len(feas_ells)
            var = sum((x - mu) ** 2 for x in feas_ells) / (len(feas_ells))
            sigma = math.sqrt(var) + eps
        else:
            mu = feas_ells[0] if len(feas_ells) == 1 else 0.0
            sigma = 1.0

        for i in feasible_idx:
            if not math.isfinite(ell[i]):
                r_dist[i] = 0.0
                continue
            hat = (ell[i] - mu) / (sigma + eps)
            r_dist[i] = 1.0 / (1.0 + math.exp(hat))  # 你们公式

    # ---- diversity：只在 S_x^+ 上算边际贡献
    r_div = [0.0] * K
    if len(feasible_idx) >= 2:
        embs = []
        for i in feasible_idx:
            embs.append(_embed_text(solutions[i]))

        D_full = _set_diversity(embs)

        for pos, i in enumerate(feasible_idx):
            embs_minus = embs[:pos] + embs[pos+1:]
            D_minus = _set_diversity(embs_minus)
            delta = D_full - D_minus
            r_div[i] = max(0.0, float(delta))

    # ---- gated total reward
    for i in range(K):
        if r_task[i] == 0:
            rewards[i] = 0.0
        else:
            rewards[i] = float(
                r_task[i]
                + r_task[i] * (lambda_dist * r_dist[i] + lambda_div * r_div[i])
            )
    return rewards

# ===================== 7) verl 入口（既支持单条，也支持组内 list） =====================
def compute_rewrite_reward(
    data_source: str,
    solution_str: Any,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """
    - 如果 solution_str 是 list[str]：按组返回 list[float]
    - 否则：退化成单样本（无法做组内归一化/多样性，就只返回 task gate + dist(不归一))
    """
    if not str(data_source).startswith("our-task/numinamath-rewrite"):
        return 0.0 if not isinstance(solution_str, list) else [0.0] * len(solution_str)

    question = ""
    if isinstance(extra_info, dict):
        question = extra_info.get("question_used") or extra_info.get("problem") or ""

    # 你们论文里的超参
    lambda_dist = float(kwargs.get("lambda_dist", 0.35))
    lambda_div  = float(kwargs.get("lambda_div", 0.5))
    enable_judge = bool(kwargs.get("enable_reasoning_judge", True))

    if isinstance(solution_str, list):
        return compute_group_rewards_for_one_prompt(
            question=question,
            solutions=solution_str,
            ground_truth=ground_truth,
            lambda_dist=lambda_dist,
            lambda_div=lambda_div,
            enable_reasoning_judge=enable_judge,
        )

    # ---- 单样本退化：没有组就无法 whiten，也无法边际多样性
    pred = normalize(extract_boxed_or_last_number(solution_str))
    gold = normalize(extract_boxed_or_last_number(ground_truth))
    v_ans = 1 if (pred is not None and gold is not None and pred == gold) else 0

    if v_ans == 0:
        return 0.0

    v_rea = 1
    if enable_judge and question and gold is not None:
        v_rea = _judge_reasoning_yesno(question=question, gold_boxed=gold, solution=solution_str)

    r_task = v_ans * v_rea
    if r_task == 0:
        return 0.0

    ell = _length_norm_nll_under_pi0(question=question, solution=solution_str)
    # 没有组内 mu/sigma，只能把 r_dist 当成一个单调映射（弱一些）
    r_dist = 1.0 / (1.0 + math.exp((ell - ell)))  # = 0.5
    # 单样本不给 diversity
    return float(r_task + r_task * (lambda_dist * r_dist))

my_reward_function = compute_rewrite_reward
