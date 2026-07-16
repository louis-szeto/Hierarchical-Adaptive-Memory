.PHONY: help install install-hf test smoke smoke-figures longmemeval locomo synthetic report arch-demo poc-real clean

PY ?= python
CONFIG ?= configs/smoke.yaml
OUT ?= results/smoke

help:
	@echo "Targets:"
	@echo "  install        Install core package (mock backend, synthetic benchmark)"
	@echo "  install-hf     Install with real HF backend + faiss + zstd + plotting extras"
	@echo "  test           Run the unit + smoke-pipeline test suite"
	@echo "  smoke          Run the tiny end-to-end smoke experiment (mock backend)"
	@echo "  smoke-figures  Regenerate tables + figures from the smoke run (watermarked)"
	@echo "  synthetic      Run the deterministic synthetic multi-session benchmark"
	@echo "  longmemeval    Run LongMemEval (requires the datasets extra + local data)"
	@echo "  locomo         Run LoCoMo (requires local locomo10.json)"
	@echo "  report         Build paper-ready tables/figures from a results dir (OUT=...)"
	@echo "  arch-demo      Run the optional stage-F HAM-layer toy integration (needs torch)"
	@echo "  poc-real       Small REAL-MODEL proof of concept (SmolLM2-135M-Instruct, CPU)"
	@echo "  clean          Remove build/test caches"

install:
	$(PY) -m pip install -e .

install-hf:
	$(PY) -m pip install -e ".[hf,faiss,zstd,datasets,plot,instr,dev]"

test:
	$(PY) -m pytest

smoke:
	$(PY) -m ham.cli run --config $(CONFIG) --out $(OUT)

smoke-figures: smoke
	$(PY) -m ham.cli report --run-dir $(OUT) --out $(OUT)/paper_artifacts

synthetic:
	$(PY) -m ham.cli run --config configs/synthetic.yaml --out results/synthetic

longmemeval:
	$(PY) -m ham.cli run --config configs/longmemeval.yaml --out results/longmemeval

locomo:
	$(PY) -m ham.cli run --config configs/locomo.yaml --out results/locomo

report:
	$(PY) -m ham.cli report --run-dir $(OUT) --out $(OUT)/paper_artifacts

arch-demo:
	$(PY) -m ham.cli arch-demo

poc-real:
	$(PY) -m ham.cli run    --config configs/poc_real_smollm.yaml --out results/poc_real_smollm
	$(PY) -m ham.cli report --run-dir results/poc_real_smollm --out results/poc_real_smollm/artifacts

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist
