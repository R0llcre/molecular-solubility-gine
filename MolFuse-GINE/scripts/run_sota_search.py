from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import yaml

PYTHON = sys.executable or "python3"


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _run(cmd: list[str], dry_run: bool, skip_if_exists: Path | None = None) -> None:
    if skip_if_exists is not None and skip_if_exists.exists():
        if dry_run:
            print(f"# resume existing {skip_if_exists}")
            return
        print(f"resume existing {skip_if_exists}")
        return
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _with_overwrite(cmd: list[str], overwrite: bool) -> list[str]:
    return [*cmd, "--overwrite"] if overwrite else cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a MolFuse SOTA search plan YAML.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)

    seeds = ",".join(_as_list(plan.get("seeds", "1,2,3,4,5")))
    seed_values = _as_list(plan.get("seeds", "1,2,3,4,5"))
    root = Path(plan.get("output_root", "sota_runs_v2"))
    configs = _as_list(plan.get("configs"))
    if not configs:
        raise ValueError("Search plan must define configs")

    if "split_audit" in plan:
        audit = plan["split_audit"] or {}
        audit_cmd = [
            PYTHON,
            "scripts/audit_splits.py",
            "--configs",
            ",".join(configs),
            "--seeds",
            seeds,
            "--out",
            str(audit.get("out", "reports/split_audit")),
        ]
        if audit.get("actual_split_template"):
            audit_cmd.extend(["--actual-split-template", str(audit.get("actual_split_template"))])
        _run(
            audit_cmd,
            args.dry_run,
        )

    for cfg in configs:
        name = Path(cfg).stem
        cfg_out = root / name
        if "graph_variants" in plan:
            _run(
                _with_overwrite(
                    [
                    PYTHON,
                    "scripts/run_ablation.py",
                    "--config",
                    cfg,
                    "--variants",
                    ",".join(_as_list(plan["graph_variants"])),
                    "--seeds",
                    seeds,
                    "--out",
                    str(cfg_out / "graph_ablation"),
                    ],
                    args.overwrite,
                ),
                args.dry_run,
            )
        if "tabular" in plan:
            tab = plan["tabular"]
            tab_out_name = str(tab.get("out_name", "tabular"))
            _run(
                _with_overwrite(
                    [
                    PYTHON,
                    "scripts/run_tabular_baselines.py",
                    "--config",
                    cfg,
                    "--feature-sets",
                    ",".join(_as_list(tab.get("feature_sets"))),
                    "--models",
                    ",".join(_as_list(tab.get("models"))),
                    "--seeds",
                    seeds,
                    "--out",
                    str(cfg_out / tab_out_name),
                    ],
                    args.overwrite,
                ),
                args.dry_run,
            )
        if "tabular_stack" in plan:
            stack = plan["tabular_stack"]
            stack_out_name = str(stack.get("out_name", "tabular_stack"))
            _run(
                _with_overwrite(
                    [
                    PYTHON,
                    "scripts/run_oof_stack.py",
                    "--config",
                    cfg,
                    "--experts",
                    ",".join(_as_list(stack.get("experts"))),
                    "--meta-model",
                    str(stack.get("meta_model", "ridge")),
                    "--folds",
                    str(stack.get("folds", 5)),
                    "--seeds",
                    seeds,
                    "--out",
                    str(cfg_out / stack_out_name),
                    ],
                    args.overwrite,
                ),
                args.dry_run,
            )
        if "graph_oof_variants" in plan:
            _run(
                _with_overwrite(
                    [
                    PYTHON,
                    "scripts/run_graph_oof_experts.py",
                    "--config",
                    cfg,
                    "--variants",
                    ",".join(_as_list(plan["graph_oof_variants"])),
                    "--seeds",
                    seeds,
                    "--folds",
                    str(plan.get("graph_oof_folds", 5)),
                    "--out",
                    str(cfg_out / "graph_oof"),
                    ],
                    args.overwrite,
                ),
                args.dry_run,
            )
        if "pretrained_oof" in plan:
            pre = plan["pretrained_oof"]
            pre_out_name = str(pre.get("out_name", "pretrained_oof"))
            cmd = [
                    PYTHON,
                    "scripts/run_pretrained_smiles_oof.py",
                    "--config",
                    cfg,
                    "--model-names",
                    ",".join(_as_list(pre.get("model_names"))),
                    "--head-models",
                    ",".join(_as_list(pre.get("head_models", "ridge"))),
                    "--random-smiles",
                    str(pre.get("random_smiles", 16)),
                    "--folds",
                    str(pre.get("folds", 5)),
                    "--batch-size",
                    str(pre.get("batch_size", 32)),
                    "--device",
                    str(pre.get("device", "auto")),
                    "--seeds",
                    seeds,
                    "--out",
                    str(cfg_out / pre_out_name),
            ]
            if pre.get("feature_sets") is not None:
                cmd.extend(["--feature-sets", ",".join(_as_list(pre.get("feature_sets")))])
            _run(
                _with_overwrite(cmd, args.overwrite),
                args.dry_run,
            )
        if "mixed_stack" in plan:
            mixed = plan["mixed_stack"]
            meta_models = _as_list(mixed.get("meta_models", "ridge"))
            templates = _as_list(mixed.get("expert_dir_templates"))
            mixed_out_name = str(mixed.get("out_name", "mixed_stack"))
            stack_name = str(mixed.get("name", mixed_out_name))
            if not templates:
                templates = [
                    "{cfg_out}/tabular_stack/runs/seed{seed}_ridge",
                    "{cfg_out}/graph_oof/runs/seed{seed}_gine_desc_concat",
                ]
            for seed in seed_values:
                for meta_model in meta_models:
                    expert_dirs = [
                        template.format(cfg_out=cfg_out, name=name, seed=seed)
                        for template in templates
                    ]
                    _run(
                        _with_overwrite(
                            [
                            PYTHON,
                            "scripts/run_mixed_stack.py",
                            "--expert-dirs",
                            ",".join(expert_dirs),
                            "--meta-model",
                            meta_model,
                            "--seed",
                            str(seed),
                            "--stack-name",
                            stack_name,
                            "--out",
                            str(cfg_out / mixed_out_name / "runs" / f"seed{seed}_{meta_model}"),
                            ],
                            args.overwrite,
                        ),
                        args.dry_run,
                        skip_if_exists=(
                            None
                            if args.overwrite
                            else cfg_out
                            / mixed_out_name
                            / "runs"
                            / f"seed{seed}_{meta_model}"
                            / "mixed_stack_summary.csv"
                        ),
                    )
        if "pruned_mixed_stack" in plan:
            pruned = plan["pruned_mixed_stack"]
            meta_models = _as_list(pruned.get("meta_models", "ridge"))
            templates = _as_list(pruned.get("expert_dir_templates"))
            pruned_out_name = str(pruned.get("out_name", "pruned_mixed_stack"))
            stack_name = str(pruned.get("name", pruned_out_name))
            if not templates:
                templates = [
                    "{cfg_out}/tabular_stack/runs/seed{seed}_ridge",
                    "{cfg_out}/graph_oof/runs/seed{seed}_gine_desc_concat",
                ]
            for seed in seed_values:
                for meta_model in meta_models:
                    expert_dirs = [
                        template.format(cfg_out=cfg_out, name=name, seed=seed)
                        for template in templates
                    ]
                    cmd = [
                        PYTHON,
                        "scripts/run_pruned_mixed_stack.py",
                        "--expert-dirs",
                        ",".join(expert_dirs),
                        "--meta-model",
                        meta_model,
                        "--seed",
                        str(seed),
                        "--stack-name",
                        stack_name,
                        "--top-k",
                        str(pruned.get("top_k", 6)),
                        "--max-experts",
                        str(pruned.get("max_experts", 10)),
                        "--corr-threshold",
                        str(pruned.get("corr_threshold", 0.95)),
                        "--oof-rmse-slack",
                        str(pruned.get("oof_rmse_slack", 0.15)),
                        "--min-existing-experts",
                        str(pruned.get("min_existing_experts", 1)),
                        "--out",
                        str(
                            cfg_out
                            / pruned_out_name
                            / "runs"
                            / f"seed{seed}_{meta_model}"
                        ),
                    ]
                    include_patterns = ",".join(_as_list(pruned.get("include_patterns")))
                    exclude_patterns = ",".join(_as_list(pruned.get("exclude_patterns")))
                    if include_patterns:
                        cmd.extend(["--include-patterns", include_patterns])
                    if exclude_patterns:
                        cmd.extend(["--exclude-patterns", exclude_patterns])
                    if pruned.get("allow_missing_experts", False):
                        cmd.append("--allow-missing-experts")
                    _run(
                        _with_overwrite(cmd, args.overwrite),
                        args.dry_run,
                        skip_if_exists=(
                            None
                            if args.overwrite
                            else cfg_out
                            / pruned_out_name
                            / "runs"
                            / f"seed{seed}_{meta_model}"
                            / "pruned_stack_summary.csv"
                        ),
                    )
        if "distill" in plan:
            distill = plan["distill"]
            variants = _as_list(distill.get("variants", "gine_desc_concat"))
            weights = _as_list(distill.get("weights", "0.5"))
            teacher_template = str(
                distill.get(
                    "teacher_template",
                    "{cfg_out}/mixed_stack/runs/seed{seed}_ridge",
                )
            )
            split_template = str(
                distill.get(
                    "split_template",
                    "{cfg_out}/tabular_stack/splits/seed{seed}",
                )
            )
            for seed in seed_values:
                for variant in variants:
                    for weight in weights:
                        _run(
                            _with_overwrite(
                                [
                                PYTHON,
                                "scripts/run_distill.py",
                                "--base-config",
                                cfg,
                                "--split-dir",
                                split_template.format(cfg_out=cfg_out, name=name, seed=seed),
                                "--teacher-run",
                                teacher_template.format(cfg_out=cfg_out, name=name, seed=seed),
                                "--distill-weight",
                                str(weight),
                                "--variant",
                                variant,
                                "--seed",
                                str(seed),
                                "--out",
                                str(
                                    cfg_out
                                    / "distill"
                                    / f"seed{seed}_{variant}_w{str(weight).replace('.', 'p')}"
                                ),
                                ],
                                args.overwrite,
                            ),
                            args.dry_run,
                            skip_if_exists=(
                                None
                                if args.overwrite
                                else cfg_out
                                / "distill"
                                / f"seed{seed}_{variant}_w{str(weight).replace('.', 'p')}"
                                / "distill_summary.json"
                            ),
                        )


if __name__ == "__main__":
    main()
