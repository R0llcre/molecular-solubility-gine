#!/usr/bin/env bash
# Single end-to-end entrypoint: run every search round under ONE output root and
# emit the leaderboard + diagnostics. The four plans already cover graph/tabular
# experts, OOF stacks, graph OOF, pretrained, mixed/pruned stacking and
# distillation, so this is the whole pipeline.
#
#   bash scripts/run_all.sh                 # full run into sota_runs_v2/
#   EXTRA_ARGS=--dry-run bash scripts/run_all.sh   # print the plan, run nothing
#   ROOT=my_runs bash scripts/run_all.sh    # use a different output root
set -euo pipefail

ROOT="${ROOT:-sota_runs_v2}"
REPORTS="${REPORTS:-reports/sota_search}"
BASELINE="${BASELINE:-gine_desc}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. --dry-run or --overwrite

PLANS=(
  "configs/search/round1_mixed_stack.yaml"
  "configs/search/round2_pretrained_fragment_3d.yaml"
  "configs/search/round3_pruned_mixed_stack.yaml"
  "configs/search/round4_pretrained_fusion.yaml"
)

for plan in "${PLANS[@]}"; do
  echo "[run_all] ${plan} -> ${ROOT}"
  # shellcheck disable=SC2086
  python scripts/run_sota_search.py --plan "${plan}" --output-root "${ROOT}" ${EXTRA_ARGS}
done

case " ${EXTRA_ARGS} " in
  *" --dry-run "*)
    # The plans only printed commands; generating reports here would overwrite
    # ${REPORTS} with an empty leaderboard, so skip it during a dry run.
    echo "[run_all] dry-run: skipping leaderboard/diagnostics generation"
    ;;
  *)
    echo "[run_all] leaderboard + diagnostics -> ${REPORTS}"
    python scripts/summarize_sota_results.py --root "${ROOT}" --out "${REPORTS}"
    python scripts/analyze_expert_pool.py --root "${ROOT}" --out "${REPORTS}_diagnostics" --baseline "${BASELINE}"
    echo "[run_all] done. Headline leaderboard: ${REPORTS}/sota_system_report.md"
    ;;
esac
