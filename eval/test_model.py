#!/usr/bin/env python3
"""
test_model.py — Interactive qualitative testing for Mythos v3

Runs a curated set of real-world cybersecurity prompts through the fine-tuned
model (and optionally the base model side-by-side) so you can judge output
quality directly rather than relying on keyword scoring.

Modes
─────
  auto      — run all 8 built-in prompts (one per task type), print full responses
  compare   — run auto prompts through BOTH models side-by-side (takes longer)
  chat      — interactive REPL: type prompts, get responses

Usage
─────
  # Single model, all built-in prompts
  python3 eval/test_model.py --adapter ./mythos-v3-7b-lora/final

  # Side-by-side comparison vs base
  python3 eval/test_model.py --adapter ./mythos-v3-7b-lora/final --compare

  # Interactive chat
  python3 eval/test_model.py --adapter ./mythos-v3-7b-lora/final --chat

  # Run just one specific task type
  python3 eval/test_model.py --adapter ./mythos-v3-7b-lora/final --task exploit_generation

  # Merged model
  python3 eval/test_model.py --model ./mythos-v3-merged --compare
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

SYSTEM_PROMPT = (
    "You are Mythos, an expert cybersecurity AI assistant specializing in "
    "vulnerability research, exploit development, and penetration testing. "
    "Provide detailed, technical, and actionable responses."
)

# ─── One representative prompt per task type ─────────────────────────────────
# These are real questions a security researcher would actually ask.

BUILTIN_PROMPTS = [
    {
        "task": "cve_analysis",
        "label": "CVE Analysis — CVE-2024-21762 (FortiOS RCE)",
        "prompt": (
            "Analyze CVE-2024-21762. What is the vulnerability class, "
            "which component is affected, and what does a working exploit chain look like? "
            "Include the CVSS score and whether public PoC exists."
        ),
    },
    {
        "task": "exploit_generation",
        "label": "Exploit Generation — SQLi PoC",
        "prompt": (
            "Write a Python script demonstrating time-based blind SQL injection "
            "against a login endpoint at POST /api/login with JSON body "
            "{\"username\": ..., \"password\": ...}. "
            "The backend is MySQL. Extract the admin password character by character."
        ),
    },
    {
        "task": "scanning_strategy",
        "label": "Scanning Strategy — Web App Pentest",
        "prompt": (
            "I'm starting a black-box penetration test of a web application. "
            "The target stack is React + Node.js/Express + PostgreSQL. "
            "Give me a step-by-step recon and testing methodology, "
            "including the specific tools and commands you would run at each stage."
        ),
    },
    {
        "task": "bug_bounty_report",
        "label": "Bug Bounty Report — SSRF to AWS Metadata",
        "prompt": (
            "I found an SSRF vulnerability on a bug bounty program. "
            "The endpoint /api/fetch?url= makes server-side HTTP requests. "
            "I was able to reach http://169.254.169.254/latest/meta-data/iam/security-credentials/ "
            "and read IAM credentials. Write a professional bug bounty report for this finding."
        ),
    },
    {
        "task": "scope_analysis",
        "label": "Scope Analysis — Wildcard boundary",
        "prompt": (
            "A bug bounty program has this scope: "
            "IN SCOPE: *.company.com, company.io, Android/iOS apps. "
            "OUT OF SCOPE: legacy.company.com, third-party integrations. "
            "I found: (1) Stored XSS on mail.company.com, "
            "(2) SSRF on api.company.com pointing to legacy.company.com's internal API, "
            "(3) RCE on a third-party payment processor used by company.com. "
            "Which of these are in scope and reportable?"
        ),
    },
    {
        "task": "tool_chaining",
        "label": "Tool Chaining — Full recon pipeline",
        "prompt": (
            "Show me a complete automated recon pipeline for a bug bounty target 'example.com' "
            "using subfinder, httpx, nuclei, and gau. "
            "Write the exact bash commands, explain what each tool does, "
            "and show how to pipe the output from one tool into the next."
        ),
    },
    {
        "task": "target_adaptation",
        "label": "Target Adaptation — IMDSv2 SSRF bypass",
        "prompt": (
            "I have an SSRF exploit that reads AWS metadata from http://169.254.169.254/latest/meta-data/. "
            "The target enforces IMDSv2 which requires a PUT request to get a token first. "
            "My current SSRF only supports GET requests. "
            "How do I adapt my attack to work with IMDSv2?"
        ),
    },
    {
        "task": "exploit_debugging",
        "label": "Exploit Debugging — Buffer overflow crashes",
        "prompt": (
            "My buffer overflow exploit controls EIP (value: 0x41414141) but crashes "
            "with SIGSEGV when the shellcode tries to execute. "
            "GDB shows ESP points directly to my shellcode. "
            "NX/DEP is enabled on this 64-bit binary. "
            "What are all the possible reasons it's crashing and how do I fix each one?"
        ),
    },
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

WIDTH = 100

def _hr(char="─", label=""):
    if label:
        side = (WIDTH - len(label) - 2) // 2
        return char * side + f" {label} " + char * (WIDTH - side - len(label) - 2)
    return char * WIDTH


def _wrap(text: str, indent: int = 2) -> str:
    lines = text.split("\n")
    wrapped = []
    prefix = " " * indent
    for line in lines:
        if len(line) <= WIDTH - indent:
            wrapped.append(prefix + line)
        else:
            for chunk in textwrap.wrap(line, width=WIDTH - indent):
                wrapped.append(prefix + chunk)
    return "\n".join(wrapped)


def _print_response(label: str, prompt: str, response: str, elapsed: float):
    print()
    print(_hr("═", label))
    print()
    print("  PROMPT:")
    print(_wrap(prompt, indent=4))
    print()
    print(f"  RESPONSE  ({elapsed:.1f}s, {len(response.split())} words):")
    print(_hr())
    print(_wrap(response, indent=2))
    print(_hr())


def _print_comparison(
    label: str,
    prompt: str,
    ft_resp: str,
    ft_time: float,
    base_resp: str,
    base_time: float,
):
    print()
    print(_hr("═", label))
    print()
    print("  PROMPT:")
    print(_wrap(prompt, indent=4))
    print()
    print(_hr("-", f"▶ FINE-TUNED  ({ft_time:.1f}s, {len(ft_resp.split())} words)"))
    print(_wrap(ft_resp, indent=2))
    print()
    print(_hr("-", f"▶ BASE MODEL  ({base_time:.1f}s, {len(base_resp.split())} words)"))
    print(_wrap(base_resp, indent=2))
    print(_hr())


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_path: str, adapter_path: str | None, gpu_util: float = 0.88):
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer

    tp = max(1, torch.cuda.device_count())

    if adapter_path:
        print(f"  Loading: {model_path} + LoRA {Path(adapter_path).name}")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            dtype="bfloat16",
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=False,
            enable_lora=True,
            max_lora_rank=128,
        )
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("mythos", 1, adapter_path)
        tok_path = adapter_path
    else:
        print(f"  Loading: {model_path}")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            dtype="bfloat16",
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=False,
        )
        lora_req = None
        tok_path = model_path

    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    print("  Ready.\n")
    return llm, tok, lora_req


def generate(llm, tok, lora_req, prompt: str, max_tokens: int = 800) -> tuple[str, float]:
    from vllm import SamplingParams

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(temperature=0.3, max_tokens=max_tokens, top_p=0.9)

    t0 = time.time()
    outputs = (
        llm.generate([formatted], params, lora_request=lora_req)
        if lora_req
        else llm.generate([formatted], params)
    )
    elapsed = time.time() - t0
    return outputs[0].outputs[0].text.strip(), elapsed


def destroy(llm):
    import gc, torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()


# ─── Modes ───────────────────────────────────────────────────────────────────

def mode_auto(args):
    prompts = [p for p in BUILTIN_PROMPTS if not args.task or p["task"] == args.task]
    if not prompts:
        print(f"Unknown task: {args.task}")
        sys.exit(1)

    print(_hr("═", "Loading model"))
    llm, tok, lora_req = load_model(args.model, args.adapter)

    for p in prompts:
        resp, elapsed = generate(llm, tok, lora_req, p["prompt"], args.max_tokens)
        _print_response(p["label"], p["prompt"], resp, elapsed)

    destroy(llm)

    if args.save:
        _save_results(args.save, prompts, [(p["prompt"], None) for p in prompts])


def mode_compare(args):
    prompts = [p for p in BUILTIN_PROMPTS if not args.task or p["task"] == args.task]

    # Run fine-tuned model
    print(_hr("═", "Loading FINE-TUNED model"))
    ft_llm, ft_tok, ft_lora = load_model(args.model, args.adapter)
    ft_results = []
    for p in prompts:
        resp, elapsed = generate(ft_llm, ft_tok, ft_lora, p["prompt"], args.max_tokens)
        ft_results.append((resp, elapsed))
        print(f"  ✓ {p['label'][:60]}")
    destroy(ft_llm)

    # Run base model
    print(_hr("═", "Loading BASE model"))
    base_llm, base_tok, _ = load_model(args.base, None)
    base_results = []
    for p in prompts:
        resp, elapsed = generate(base_llm, base_tok, None, p["prompt"], args.max_tokens)
        base_results.append((resp, elapsed))
        print(f"  ✓ {p['label'][:60]}")
    destroy(base_llm)

    # Print side-by-side
    print("\n\n")
    print(_hr("═", "COMPARISON RESULTS"))
    for p, (ft_r, ft_t), (base_r, base_t) in zip(prompts, ft_results, base_results):
        _print_comparison(p["label"], p["prompt"], ft_r, ft_t, base_r, base_t)

    if args.save:
        path = Path(args.save)
        with open(path, "w") as f:
            for p, (ft_r, _), (base_r, _) in zip(prompts, ft_results, base_results):
                f.write(f"## {p['label']}\n\n")
                f.write(f"**Prompt:** {p['prompt']}\n\n")
                f.write(f"**Fine-tuned:**\n{ft_r}\n\n")
                f.write(f"**Base:**\n{base_r}\n\n")
                f.write("---\n\n")
        print(f"\n  Saved → {path}")


def mode_chat(args):
    print(_hr("═", "Interactive Chat — Mythos v3"))
    print("  Type your prompt and press Enter twice.")
    print("  Commands: :exit  :task <name>  :base (toggle base comparison)")
    print()

    llm, tok, lora_req = load_model(args.model, args.adapter)

    compare_base = False
    base_llm = base_tok = None

    def _ensure_base():
        nonlocal compare_base, base_llm, base_tok
        if not compare_base:
            return
        if base_llm is None:
            print("  Loading base model for comparison...")
            base_llm, base_tok, _ = load_model(args.base, None)

    try:
        while True:
            print("\n" + _hr("─", "Your prompt  (blank line to submit, :exit to quit)"))
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line == ":exit":
                    raise KeyboardInterrupt
                if line.startswith(":task "):
                    task = line.split(None, 1)[1].strip()
                    match = [p for p in BUILTIN_PROMPTS if p["task"] == task]
                    if match:
                        lines = [match[0]["prompt"]]
                        print(f"  [Using built-in prompt for: {task}]")
                        break
                    else:
                        tasks = [p["task"] for p in BUILTIN_PROMPTS]
                        print(f"  Available tasks: {', '.join(tasks)}")
                    continue
                if line == ":base":
                    compare_base = not compare_base
                    print(f"  Base comparison: {'ON' if compare_base else 'OFF'}")
                    continue
                if line == "" and lines:
                    break
                if line:
                    lines.append(line)

            if not lines:
                continue

            prompt = "\n".join(lines)
            print()

            ft_resp, ft_time = generate(llm, tok, lora_req, prompt, args.max_tokens)
            print(_hr("-", f"▶ FINE-TUNED  ({ft_time:.1f}s)"))
            print(_wrap(ft_resp, indent=2))

            if compare_base:
                _ensure_base()
                base_resp, base_time = generate(base_llm, base_tok, None, prompt, args.max_tokens)
                print()
                print(_hr("-", f"▶ BASE MODEL  ({base_time:.1f}s)"))
                print(_wrap(base_resp, indent=2))

            print(_hr())

    except KeyboardInterrupt:
        print("\n  Exiting.")

    finally:
        destroy(llm)
        if base_llm:
            destroy(base_llm)


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Mythos v3 qualitative model tester")

    # Model specification (same pattern as eval.py)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--adapter", help="LoRA adapter dir (e.g. ./mythos-v3-7b-lora/final)")
    grp.add_argument("--model",   help="Full merged model path or HF model ID",
                     default="Qwen/Qwen2.5-7B-Instruct")

    p.add_argument("--base",      default="Qwen/Qwen2.5-7B-Instruct",
                   help="Base model for comparison")

    # Mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--compare", action="store_true",
                      help="Run all prompts through BOTH models side-by-side")
    mode.add_argument("--chat",    action="store_true",
                      help="Interactive REPL mode")

    p.add_argument("--task", default=None,
                   choices=[p["task"] for p in BUILTIN_PROMPTS] + [None],
                   help="Run only one specific task type")
    p.add_argument("--max-tokens", type=int, default=800,
                   help="Max tokens per response (default: 800)")
    p.add_argument("--save",  default=None,
                   help="Save responses to this file (.md format)")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect adapter if neither --adapter nor --model is explicitly set
    if args.adapter is None and args.model == "Qwen/Qwen2.5-7B-Instruct":
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        for candidate in [
            project_dir / "mythos-v3-7b-merged",
            project_dir / "mythos-v3-merged",
        ]:
            if candidate.exists():
                args.model = str(candidate)
                args.adapter = None
                break
        else:
            for candidate in [
                project_dir / "mythos-v3-7b-lora" / "final",
                project_dir / "mythos-v3-lora"    / "final",
            ]:
                if candidate.exists():
                    args.adapter = str(candidate)
                    args.model   = args.base
                    break

    print()
    print("=" * WIDTH)
    print("  Mythos v3 — Qualitative Model Tester")
    print("=" * WIDTH)
    if args.adapter:
        print(f"  Mode      : LoRA adapter")
        print(f"  Adapter   : {args.adapter}")
        print(f"  Backbone  : {args.model}")
    else:
        print(f"  Model     : {args.model}")
    if args.compare or (args.chat and True):
        print(f"  Base      : {args.base}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * WIDTH)
    print()

    if args.compare:
        mode_compare(args)
    elif args.chat:
        mode_chat(args)
    else:
        mode_auto(args)


if __name__ == "__main__":
    main()
