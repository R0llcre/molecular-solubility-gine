# MolFuse-GINE convenience targets. ROOT/REPORTS can be overridden:
#   make sota ROOT=my_runs
ROOT ?= sota_runs_v2
REPORTS ?= reports/sota_search
BASELINE ?= gine_desc

.PHONY: help install test smoke sota report clean

help:
	@echo "Targets:"
	@echo "  install  pip install -r requirements.txt"
	@echo "  test     compileall + pytest"
	@echo "  smoke    dry-run the first search round (no compute)"
	@echo "  sota     full end-to-end search into ROOT=$(ROOT) + leaderboard"
	@echo "  report   regenerate leaderboard/diagnostics from ROOT=$(ROOT)"

install:
	pip install -r requirements.txt

test:
	python -m compileall -q molfuse_gine scripts tests
	python -m pytest -q

smoke:
	python scripts/run_sota_search.py --plan configs/search/round1_mixed_stack.yaml \
		--output-root $(ROOT) --dry-run

sota:
	ROOT=$(ROOT) REPORTS=$(REPORTS) BASELINE=$(BASELINE) bash scripts/run_all.sh

report:
	python scripts/summarize_sota_results.py --root $(ROOT) --out $(REPORTS)
	python scripts/analyze_expert_pool.py --root $(ROOT) --out $(REPORTS)_diagnostics \
		--baseline $(BASELINE)
