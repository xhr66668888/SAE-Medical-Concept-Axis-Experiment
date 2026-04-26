#!/usr/bin/env python3
"""Step 5: causal steering experiment for the diabetes type axis.

Loads the axis tensor saved by Step 3, then for each held-out prompt:

  resid_post  ←  resid_post + alpha * axis     (at the captured layer)

and measures how the logit difference between the Type 1-coded drug
("insulin") and the Type 2-coded drug ("metformin") at the final token shifts
across alphas.

The prediction (Assistant-Axis style) is that adding the axis pushes toward
Type 1 (more insulin), and subtracting it pushes toward Type 2 (more
metformin), monotonically in alpha. If the curve is flat or non-monotonic,
the axis is correlational — not causal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_med.data_utils import read_prompts, write_csv_dicts
from sae_med.model_utils import (
    PRESETS,
    bootstrap_ci,
    choose_device,
    choose_dtype,
    configure_torch,
    load_model,
    print_run_header,
    require_runtime_deps,
    resolve_token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steer along the Type 1 - Type 2 axis and measure logit shifts.")
    parser.add_argument("--prompts", default="data/diabetes_contrastive_prompts.csv")
    parser.add_argument("--axis-path", default="outputs/axis/concept_axis.pt")
    parser.add_argument("--output-dir", default="outputs/steering")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gemma3-1b-it-res")
    parser.add_argument("--model-name", help="Override model name from preset.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--use-split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--filter-complication", default="none", help="Restrict steering eval to one complication. 'all' to keep all.")
    parser.add_argument("--max-prompts", type=int, default=24)
    parser.add_argument(
        "--alphas",
        default="-3,-2,-1.5,-1,-0.5,-0.25,0,0.25,0.5,1,1.5,2,3",
        help="Comma-separated steering coefficients applied to the unscaled axis vector.",
    )
    parser.add_argument(
        "--positions",
        default="all",
        choices=["all", "last", "after_keyword"],
        help="Where to inject the steering vector. 'all' adds to every position; 'last' only the final token; "
        "'after_keyword' from the first occurrence of --keyword onward.",
    )
    parser.add_argument(
        "--keyword",
        default="diabetes",
        help="Token substring used by --positions after_keyword. Falls back to the last token if absent.",
    )
    parser.add_argument(
        "--bootstrap-trials",
        type=int,
        default=1000,
        help="Bootstrap trials for the steering-curve CI band. 0 disables.",
    )
    parser.add_argument(
        "--type1-token",
        default=" insulin",
        help="Token whose logit represents the Type 1-coded drug. Leading space matters for BPE.",
    )
    parser.add_argument(
        "--type2-token",
        default=" metformin",
        help="Token whose logit represents the Type 2-coded drug. Leading space matters for BPE.",
    )
    parser.add_argument("--no-prepend-bos", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def require_plot_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name or str(exc)}\n"
            "Install dependencies first, e.g. `pip install -r requirements.txt`."
        ) from exc
    return plt


def steering_positions(model, tokens, mode: str, keyword: str) -> tuple[list[int], str]:
    seq_len = int(tokens.shape[1])
    if mode == "all":
        return list(range(seq_len)), "all"
    if mode == "last":
        return [seq_len - 1], "last"
    if mode != "after_keyword":
        raise ValueError(f"Unsupported steering position mode: {mode}")

    str_tokens = model.to_str_tokens(tokens[0])
    needle = (keyword or "").lower()
    if needle:
        for idx, token in enumerate(str_tokens):
            if needle in token.lower():
                return list(range(idx, seq_len)), f"after_keyword:{idx}"
    return [seq_len - 1], "after_keyword:fallback_last"


def main() -> None:
    args = parse_args()
    plt = require_plot_deps()
    torch, _, HookedTransformer = require_runtime_deps()
    configure_torch(torch)

    axis_path = Path(args.axis_path)
    if not axis_path.exists():
        raise SystemExit(f"Axis file not found: {axis_path}. Run Step 3 first.")
    axis_blob = torch.load(axis_path, map_location="cpu", weights_only=False)
    axis = axis_blob["axis"].float()
    hook_name = axis_blob["hook_name"]
    layer = int(axis_blob["layer"])
    saved_model_name = axis_blob.get("model_name", None)

    preset = PRESETS[args.preset]
    model_name = args.model_name or saved_model_name or preset.model_name
    if saved_model_name and saved_model_name != model_name:
        print(f"NOTE: axis was saved with model={saved_model_name}, running with {model_name}.")
    prepend_bos = not args.no_prepend_bos

    rows = read_prompts(args.prompts)
    if args.use_split != "all":
        rows = [row for row in rows if row.get("split", "train") == args.use_split]
    if args.filter_complication != "all":
        rows = [row for row in rows if row.get("complication") == args.filter_complication]
    rows = [row for row in rows if row.get("diabetes_type") in {"type1", "type2"}]
    rows = rows[: args.max_prompts] if args.max_prompts else rows
    if len(rows) == 0:
        raise SystemExit("No prompts left after filtering. Loosen --use-split / --filter-complication.")

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    if 0.0 not in alphas:
        alphas.insert(0, 0.0)
    alphas = sorted(set(alphas))

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, device, args.dtype)
    print_run_header(
        model_name=model_name,
        sae_release=None,
        device=device,
        dtype=dtype,
        extra={
            "hook_name": hook_name,
            "layer": layer,
            "prompts": len(rows),
            "alphas": alphas,
            "positions": args.positions,
            "axis_norm": float(axis.norm().item()),
        },
    )

    model = load_model(HookedTransformer, model_name, device, dtype, prepend_bos)

    type1_id = resolve_token_id(model.tokenizer, args.type1_token)
    type2_id = resolve_token_id(model.tokenizer, args.type2_token)
    print(f"type1 token id ({args.type1_token!r}): {type1_id}")
    print(f"type2 token id ({args.type2_token!r}): {type2_id}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    per_prompt_logit_diff: dict[int, dict[float, float]] = {}

    axis_on_device = axis.to(device=device, dtype=dtype)

    def make_hook(alpha: float, position_indices: list[int]):
        delta = alpha * axis_on_device

        def hook_fn(activation, hook):  # signature required by TransformerLens hooks
            if args.positions == "all":
                return activation + delta
            patched = activation.clone()
            patched[:, position_indices, :] = patched[:, position_indices, :] + delta
            return patched

        return hook_fn

    for prompt_idx, row in enumerate(rows, start=1):
        prompt = row["prompt"]
        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
        position_indices, position_label = steering_positions(model, tokens, args.positions, args.keyword)
        per_prompt_logit_diff[prompt_idx] = {}
        for alpha in alphas:
            hook_fn = make_hook(alpha, position_indices)
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                logits = model(tokens, return_type="logits")
            last_logits = logits[0, -1, :].float().detach().cpu()
            log_probs = torch.log_softmax(last_logits, dim=-1)
            type1_logit = float(last_logits[type1_id].item())
            type2_logit = float(last_logits[type2_id].item())
            type1_logp = float(log_probs[type1_id].item())
            type2_logp = float(log_probs[type2_id].item())
            top_value, top_id = last_logits.topk(1)
            top_token = model.tokenizer.decode([int(top_id.item())])
            logit_diff = type1_logit - type2_logit
            per_prompt_logit_diff[prompt_idx][alpha] = logit_diff
            result_rows.append(
                {
                    "prompt_index": prompt_idx,
                    "prompt_id": row.get("prompt_id", ""),
                    "diabetes_type": row.get("diabetes_type", ""),
                    "complication": row.get("complication", ""),
                    "split": row.get("split", ""),
                    "alpha": alpha,
                    "type1_logit": type1_logit,
                    "type2_logit": type2_logit,
                    "logit_diff_type1_minus_type2": logit_diff,
                    "type1_logprob": type1_logp,
                    "type2_logprob": type2_logp,
                    "logprob_diff_type1_minus_type2": type1_logp - type2_logp,
                    "top_token": top_token,
                    "top_token_logit": float(top_value.item()),
                    "steering_positions": position_label,
                    "steering_position_count": len(position_indices),
                    "prompt": prompt,
                }
            )
        if prompt_idx % 10 == 0 or prompt_idx == len(rows):
            print(f"steered {prompt_idx}/{len(rows)} prompts")

    write_csv_dicts(
        output_dir / "steering_results.csv",
        result_rows,
        [
            "prompt_index",
            "prompt_id",
            "diabetes_type",
            "complication",
            "split",
            "alpha",
            "type1_logit",
            "type2_logit",
            "logit_diff_type1_minus_type2",
            "type1_logprob",
            "type2_logprob",
            "logprob_diff_type1_minus_type2",
            "top_token",
            "top_token_logit",
            "steering_positions",
            "steering_position_count",
            "prompt",
        ],
    )

    by_alpha_delta: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    for prompt_idx, alpha_to_logit_diff in per_prompt_logit_diff.items():
        baseline = alpha_to_logit_diff[0.0]
        for alpha in alphas:
            by_alpha_delta[alpha].append(alpha_to_logit_diff[alpha] - baseline)
    deltas = [sum(by_alpha_delta[a]) / len(by_alpha_delta[a]) for a in alphas]
    ci_lower: list[float] = []
    ci_upper: list[float] = []
    for idx, alpha in enumerate(alphas):
        if args.bootstrap_trials > 0:
            _, lower, upper = bootstrap_ci(
                by_alpha_delta[alpha],
                n=args.bootstrap_trials,
                seed=20260425 + idx,
            )
        else:
            lower = upper = deltas[idx]
        ci_lower.append(lower)
        ci_upper.append(upper)

    plt.figure(figsize=(7, 4))
    plt.plot(alphas, deltas, marker="o", color="#2b6cb0")
    if args.bootstrap_trials > 0:
        plt.fill_between(alphas, ci_lower, ci_upper, color="#2b6cb0", alpha=0.18, label="95% bootstrap CI")
    plt.axhline(0, color="#9ca3af", linewidth=0.8)
    plt.axvline(0, color="#9ca3af", linewidth=0.8)
    plt.xlabel(f"alpha (multiplier on axis, ||axis||={float(axis.norm().item()):.2f})")
    plt.ylabel(f"Δ logit({args.type1_token.strip()}) − logit({args.type2_token.strip()})")
    plt.title("Steering curve: diabetes Type 1 − Type 2 axis")
    if args.bootstrap_trials > 0:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "steering_curve.png", dpi=180)
    plt.close()

    monotone_inc = all(b >= a - 1e-6 for a, b in zip(deltas, deltas[1:]))
    monotone_dec = all(b <= a + 1e-6 for a, b in zip(deltas, deltas[1:]))
    direction = "monotone-increasing" if monotone_inc else ("monotone-decreasing" if monotone_dec else "non-monotone")
    summary_lines = [
        "Steering Summary",
        f"axis_path: {axis_path}",
        f"hook_name: {hook_name}",
        f"layer: {layer}",
        f"axis_norm: {float(axis.norm().item()):.6f}",
        f"prompts: {len(rows)}  split={args.use_split}  complication={args.filter_complication}",
        f"alphas: {alphas}",
        f"positions: {args.positions}  keyword={args.keyword!r}",
        f"mean Δ logit_diff (type1 - type2) by alpha: " + ", ".join(f"a={a}:{d:+.3f}" for a, d in zip(alphas, deltas)),
        f"95% bootstrap CI by alpha: "
        + ", ".join(f"a={a}:[{lo:+.3f},{hi:+.3f}]" for a, lo, hi in zip(alphas, ci_lower, ci_upper)),
        f"monotonicity: {direction}",
        "Interpretation: positive slope = adding the axis increases preference for the type1 drug, "
        "consistent with a causal axis. Flat/non-monotone curve = the axis is correlational only.",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "steering_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
