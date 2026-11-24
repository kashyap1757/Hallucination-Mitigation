"""
real_llm.py — Real LLM backend (replaces SimulatedLLM)
=======================================================
Runs an instruction-tuned causal LM locally with Hugging Face `transformers`
and returns, for every prompt, N sampled answers together with the mean
token log-probability of each answer (a model-confidence signal that the
simulated LLM could not provide).

Default model: meta-llama/Llama-3.2-1B-Instruct (runs on an 8 GB GPU in bf16).
Any chat model on the Hub works, e.g. --llm Qwen/Qwen2.5-1.5B-Instruct.

Unlike SimulatedLLM, this model never sees the reference answers — it only
gets the question and (optionally) the retrieved RAG context.
"""

import re
import time
from typing import List, Optional

import env_compat  # noqa: F401  (must precede transformers)

DEFAULT_LLM = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = (
    "You are a factual question-answering assistant. Answer the question with a "
    "short, direct answer: a few words or one sentence. Do not explain."
)


class RealLLM:
    def __init__(self, model_name: str = DEFAULT_LLM, max_new_tokens: int = 40,
                 batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # Fail loudly instead of silently spilling into (very slow) shared
            # system memory when VRAM runs out (Windows WDDM behaviour).
            torch.cuda.set_per_process_memory_fraction(0.90)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"    [LLM] Loading {model_name} on {self.device} ({dtype})")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device).eval()
        self.model.generation_config.pad_token_id = self.tok.pad_token_id

    # ── prompting ────────────────────────────────────────────────────────────
    def build_prompt(self, question: str, context: Optional[str] = None) -> str:
        if context:
            user = (f"Context:\n{context}\n\nUse the context if it is relevant.\n\n"
                    f"Question: {question}\nAnswer:")
        else:
            user = f"Question: {question}\nAnswer:"
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        return self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^(answer|a)\s*:\s*", "", text, flags=re.I)
        text = text.split("\n")[0].strip()
        return text or "I don't know."

    # ── generation ───────────────────────────────────────────────────────────
    def generate(self, prompts: List[str], n: int = 5, temperature: float = 0.7,
                 top_p: float = 0.9, seed: int = 42, progress: bool = True) -> List[List[dict]]:
        """
        Returns one list per prompt with `n` dicts: {"text", "logprob"}.
        n == 1 → greedy decoding; n > 1 → temperature sampling.
        `logprob` is the mean per-token log-probability of the generated answer.
        """
        torch = self.torch
        torch.manual_seed(seed)
        sample = n > 1
        out_all = []
        t0 = time.time()
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start:start + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            gen_kwargs = dict(max_new_tokens=self.max_new_tokens, num_return_sequences=n,
                              do_sample=sample)
            if sample:
                gen_kwargs.update(temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                seqs = self.model.generate(**enc, **gen_kwargs)

            P = enc["input_ids"].shape[1]
            new_tokens = seqs[:, P:]
            eos_ids = self.model.generation_config.eos_token_id
            eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
            eos_ids.add(self.tok.pad_token_id)
            lengths = []
            for row in range(new_tokens.shape[0]):
                toks = new_tokens[row].tolist()
                lengths.append(next((i for i, t in enumerate(toks) if t in eos_ids), len(toks)))
            prompt_mask = enc["attention_mask"].repeat_interleave(n, dim=0)
            logps = self._answer_logprobs(seqs, prompt_mask, lengths)

            for row in range(new_tokens.shape[0]):
                toks = new_tokens[row].tolist()[:lengths[row]]
                text = self._clean(self.tok.decode(toks, skip_special_tokens=True))
                if row % n == 0:
                    out_all.append([])
                out_all[-1].append({"text": text, "logprob": logps[row]})
            del seqs

            if progress:
                done = min(start + self.batch_size, len(prompts))
                el = time.time() - t0
                print(f"\r      generated {done}/{len(prompts)}  "
                      f"({el:.0f}s, ETA {el / done * (len(prompts) - done):.0f}s)",
                      end="", flush=True)
        if progress:
            print()
        return out_all

    def _answer_logprobs(self, seqs, prompt_mask, lengths, sub_batch: int = 16):
        """
        Mean log p(token) of each generated answer under the raw (un-warped)
        model distribution, via one extra forward pass. Only the logits of the
        answer positions are materialised, in small sub-batches, so memory stays
        bounded (keeping per-step logits inside generate() overflows 8 GB GPUs).
        """
        torch = self.torch
        P = prompt_mask.shape[1]
        T = seqs.shape[1] - P
        ans_mask = torch.zeros((seqs.shape[0], T), dtype=prompt_mask.dtype,
                               device=seqs.device)
        for i, L in enumerate(lengths):
            ans_mask[i, :L] = 1
        mask = torch.cat([prompt_mask, ans_mask], dim=1)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = []
        for i in range(0, seqs.shape[0], sub_batch):
            sl = slice(i, i + sub_batch)
            with torch.no_grad():
                logits = self.model(input_ids=seqs[sl], attention_mask=mask[sl],
                                    position_ids=pos[sl], logits_to_keep=T + 1).logits
            # logits[:, j] predicts token P + j  (j = 0..T-1)
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            lp = lp.gather(-1, seqs[sl, P:].unsqueeze(-1)).squeeze(-1)
            for r in range(lp.shape[0]):
                L = max(lengths[i + r], 1)
                out.append(float(lp[r, :L].mean()))
            del logits, lp
        return out
