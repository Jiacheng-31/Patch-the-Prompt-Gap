# 🧩 RL-based Data Rewriting with VERL (GRPO) to Mitigate Catastrophic Forgetting

This repo implements an **RL-trained data rewriting agent** that rewrites downstream supervision *before* SFT to reduce distribution mismatch, stabilize training, and mitigate catastrophic forgetting ✨  
We use **[VERL](https://github.com/volcengine/verl)** for on-policy RL with **GRPO-style group optimization**, and train the rewriter as a lightweight **LoRA “patch”** on top of a frozen instruction-tuned base model. Downstream SFT is done with **LLaMA-Factory** (no code changes, so it is not included here).

<p align="center">
  <img src="figure/framework.pdf" alt="Framework" width="900"/>
</p>

---

## 📁 Repository Layout

```text
ACL-github/
├── data-example/                  # small examples / data format reference
├── figure/
│   └── framework.pdf              # framework figure
├── re-writing/
│   ├── rewriting_vllm.py          # apply the trained rewriter with vLLM
│   └── run.sh                     # rewriting entry script
├── test/
│   ├── test-general/              # general-domain retention / forgetting evaluation
│   └── test-math/
│       ├── dataset/               # math evaluation dataset
│       ├── eval_vllm.py           # vLLM-based evaluation
│       └── run.sh                 # math eval entry script
└── verl/
    └── verl/experimental/agent_loop/
        └── ...                    # ✅ modified: dedicate a GPU to RewardManager
````

---

## 🚀 What We Do (Short Version)

* **Stage I (RL with VERL + GRPO):** train a rewriting policy (R_\phi) as a LoRA patch on a frozen base model (\pi_0).
* **Stage II (Dataset + SFT):** build a rewritten dataset using **Generate–Verify–Fallback**, then run standard SFT on the rewritten data.

---

## 🎯 Reward (Method-aligned)

GitHub README does not reliably render LaTeX, so we show the reward in plain text.

### Gated reward (used for GRPO)

```

r = r_task + r_task * ( lambda_dist * r_dist + lambda_div * r_div )

```

- `r_task ∈ {0,1}`: **hard gate** (final answer correct + reasoning valid)
- `r_dist`: **QA-style alignment** score under the frozen base model `pi0(·|x)` (group-normalized)
- `r_div`: **diversity** among feasible rewrites in the same group (Qwen-Embedding + marginal contribution)

**Key detail:** `r_dist` and `r_div` are computed **only when `r_task = 1`** (feasible rewrites).

---

## 🧠 GRPO / Group Requirement (Important)

GRPO requires sampling **K candidates per input** (same prompt `x`), and computing group statistics inside RewardManager.

* Make sure your rollout generates **K rewrites per prompt**.
* RewardManager should **evaluate rewards per group** (e.g., pass `solution_str` as `List[str]` for one prompt) so we can:

  * normalize `r_dist` within the group, and
  * compute marginal diversity `r_div`.

---

## 🧩 Key Implementation Notes

### ✅ Reward function changes

We mainly modify the reward to match the paper:

* hard task gate (answer + reasoning),
* QA-style alignment under `pi0(y|x)` with group normalization,
* diversity via Qwen-Embedding with marginal contribution.

### ✅ Dedicated GPU for RewardManager

We modify:

```text
verl/verl/experimental/agent_loop/
```

to allocate a dedicated GPU for the RewardManager (reduces contention and improves stability).

---

## ✍️ Dataset Rewriting (vLLM)

Run rewriting:

```bash
cd re-writing
bash run.sh
```

Main entry:

* `re-writing/rewriting_vllm.py`

This runs **Generate–Verify–Fallback**:

* generate rewrite `y~`
* if `r_task=1` accept `y~`
* else fallback to expert `y*`

---

## 🧪 Evaluation

### Math evaluation

```bash
cd test/test-math
bash run.sh
```

* `test/test-math/eval_vllm.py`
* `test/test-math/dataset/`

### General retention / forgetting

* `test/test-general/`

---

## 🔧 Downstream SFT (LLaMA-Factory)

We run downstream SFT with LLaMA-Factory **without modifications**, so training code/configs are not included here.

Workflow:

1. rewrite data → build `D_R`
2. run SFT on `D_R` with LLaMA-Factory

---

## 📌 Citation

```bibtex
@article{2026rlrewriting,
  title={RL-based Data Rewriting for Stable Downstream SFT},
  author={...},
  year={2026}
}
```

---

## 🙏 Acknowledgements

* **VERL**: [https://github.com/volcengine/verl](https://github.com/volcengine/verl)
* **vLLM** for efficient generation/evaluation
* **LLaMA-Factory** for downstream SFT (used without modification)
