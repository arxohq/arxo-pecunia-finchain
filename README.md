# Arxo Pecunia FinChain 0.1

Arxo Pecunia FinChain 0.1 solves multi-step finance problems without tools: it lists what the problem gives, works through numbered steps with every calculation written out, and ends with the answer on its own line. It covers the twelve domains of FinChain, from financial statements and ratios to investment analysis, corporate finance, risk management and sustainable finance. This is `Arxo-Pecunia-4B-FinChain-0.1`, a LoRA adapter for [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).

It follows the Arxo releases on [MedCalc-Bench](https://huggingface.co/arxohq/Arxo-Materia-4B-MedCalc-0.1), [EHRSQL](https://huggingface.co/arxohq/Arxo-Materia-4B-EHRSQL-0.1) and [ChemIQ](https://huggingface.co/arxohq/Arxo-Valentia-4B-ChemIQ-0.1).

## Results

[FinChain](https://github.com/mbzuai-nlp/finchain) has no fixed test set: every question is drawn from a template with a random seed. We drew 2,800 questions, ten per template, and scored the final answer with the benchmark's own checker (ChainEval, 5% tolerance). The models answer without tools.

| System | Size | Trained on the task | Final answer correct |
|---|---|---|---|
| **Arxo Pecunia FinChain 0.1 (4B)** | 4B | yes | **2,245 / 2,800 (80.2%)** |
| Qwen3.5-4B, no training | 4B | no | 2,147 / 2,800 (76.7%) |
| **Arxo Pecunia FinChain 0.1 (0.8B)** | 0.8B | yes | **2,002 / 2,800 (71.5%)** |
| Qwen3.5-0.8B, no training | 0.8B | no | 1,265 / 2,800 (45.2%) |

One training run per model. Greedy decoding, thinking switched off, a limit of 2,048 new tokens; the same prompt and limit for the trained and the untrained rows.

For reference, the numbers the benchmark's authors report in [their paper](https://arxiv.org/abs/2506.02515) (Table 2, final-answer correctness, zero-shot). They are measured on the authors' own draw of questions, so the two tables compare the setting and not question by question.

| System | Size | Final answer correct |
|---|---|---|
| GPT-4.1 | n/a | 84.7% |
| Gemini 2.5 Pro | n/a | 84.3% |
| Claude Sonnet 4.5 | n/a | 83.3% |
| GPT-5 | n/a | 82.0% |
| Qwen 2.5 Instruct | 7B | 65.4% |
| Qwen2.5-Math | 7B | 62.6% |
| Mathstral | 7B | 54.0% |
| Fin-o1 | 8B | 52.8% |
| Fin-R1 | 7B | 52.8% |
| Qwen 3 | 8B | 32.3% |

## Results by difficulty

| Level | Questions | Arxo Pecunia (4B) | Qwen3.5-4B | Arxo Pecunia (0.8B) | Qwen3.5-0.8B |
|---|---|---|---|---|---|
| Basic | 1,130 | **1,000 (88.5%)** | 967 (85.6%) | 963 (85.2%) | 735 (65.0%) |
| Intermediate | 1,120 | **919 (82.1%)** | 854 (76.2%) | 835 (74.6%) | 400 (35.7%) |
| Advanced | 550 | 326 (59.3%) | 326 (59.3%) | 204 (37.1%) | 130 (23.6%) |
| **All** | **2,800** | **2,245 (80.2%)** | 2,147 (76.7%) | 2,002 (71.5%) | 1,265 (45.2%) |

Two notes on the numbers:

- The 2,800 questions come from 280 of the benchmark's 290 templates. The other ten templates fail to generate a question in the benchmark's code as published.
- The models were trained on questions of the same kinds as the basic and intermediate templates, written in our own wording with our own numbers. The advanced templates had no counterpart in training: there training gives the 4B model no gain, and the 0.8B model stays well below the untrained 4B model.

## What we did not do

- No test question was used in training. Of the 2,800 test questions none appears in the training set, word for word or with the numbers masked.
- The prompt was not tuned on the test questions.
- The score is not ours: the final answer is read and compared by the benchmark's checker.

## How to use

The adapter attaches to `AutoModelForImageTextToText`, not to `AutoModelForCausalLM`. The system prompt below is the one the model was trained with.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "arxohq/Arxo-Pecunia-4B-FinChain-0.1")

SYSTEM = (
    "You are a careful financial analyst. Solve the problem step by step. First list what is given, "
    "then number the steps as 'Step 1:', 'Step 2:' and so on, show each calculation, and finish with "
    "a line 'Answer = <value>' (or 'Answer: <text>' when the answer is not a number)."
)
question = (
    "Amazon has $40892 in assets and $24187 in liabilities. "
    "Using the balance sheet equation, calculate the company's equity."
)

messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

The model writes a `Given:` block, the numbered steps and a last line `Answer = 16705.00`.

## Scope

- The model is trained for the question kinds of FinChain. On other finance questions it is untested.
- It computes from the numbers stated in the question. It does not read financial statements, filings or tables, and it has no access to market data.
- It works in the plain arithmetic of the benchmark: no day-count conventions, tax rules or currency handling beyond what a question states.
- It is a research release and not financial advice.

## Files

| File | What it is |
|---|---|
| `eval/scores.jsonl` | the 2,800 questions: template, level, seed, question text, and 1 or 0 from the benchmark's checker for each of the four rows of the table |
| `adapters/4b.adapter_config.json`, `adapters/0.8b.adapter_config.json` | adapter configurations |
| `SHA256SUMS` | checksums of these files and of the adapter weights |

The adapter weights are attached to the [v0.1 release](https://github.com/arxohq/arxo-pecunia-finchain/releases/tag/v0.1) and are the same files as on Hugging Face. The questions are drawn with the benchmark's code at commit `9bd2942`, seeds 7000000 to 7000009 for every template.

## Models

- [arxohq/Arxo-Pecunia-4B-FinChain-0.1](https://huggingface.co/arxohq/Arxo-Pecunia-4B-FinChain-0.1)
- [arxohq/Arxo-Pecunia-0.8B-FinChain-0.1](https://huggingface.co/arxohq/Arxo-Pecunia-0.8B-FinChain-0.1)

FinChain is released by its authors under the Apache 2.0 license; the question in the example above is generated by its code.
