.PHONY: help install install-hf test smoke smoke-figures longmemeval synthetic report arch-demo poc-real archbench-smoke archbench-toy kvbench-smoke kvbench-real clean

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
	@echo "  report         Build paper-ready tables/figures from a results dir (OUT=...)"
	@echo "  arch-demo      Run the optional stage-F HAM-layer toy integration (needs torch)"
	@echo "  poc-real       Small REAL-MODEL proof of concept (SmolLM2-135M-Instruct, CPU)"
	@echo "  archbench-smoke Stage-F toy arch memory-block compression (mock, watermarked)"
	@echo "  archbench-toy  Stage-F toy arch memory-block compression (real torch toy model)"
	@echo "  kvbench-smoke  Stage-D KV-cache compression (mock, watermarked)"
	@echo "  kvbench-real   Stage-D KV-cache compression (frozen SmolLM2-135M, needs [hf])"
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

report:
	$(PY) -m ham.cli report --run-dir $(OUT) --out $(OUT)/paper_artifacts

arch-demo:
	$(PY) -m ham.cli arch-demo

poc-real:
	$(PY) -m ham.cli run    --config configs/poc_real_smollm.yaml --out results/poc_real_smollm
	$(PY) -m ham.cli report --run-dir results/poc_real_smollm --out results/poc_real_smollm/artifacts

archbench-smoke:
	$(PY) -m ham.cli archbench        --config configs/archbench_smoke.yaml --out results/archbench_smoke
	$(PY) -m ham.cli archbench-report --run-dir results/archbench_smoke  --out results/archbench_smoke/artifacts

archbench-toy:
	$(PY) -m ham.cli archbench        --config configs/archbench_toy.yaml --out results/archbench_toy
	$(PY) -m ham.cli archbench-report --run-dir results/archbench_toy    --out results/archbench_toy/artifacts

kvbench-smoke:
	$(PY) -m ham.cli kvbench        --config configs/kvbench_smoke.yaml --out results/kvbench_smoke
	$(PY) -m ham.cli kvbench-report --run-dir results/kvbench_smoke  --out results/kvbench_smoke/artifacts

kvbench-real:
	$(PY) -m ham.cli kvbench        --config configs/kvbench_smollm.yaml --out results/kvbench_smollm
	$(PY) -m ham.cli kvbench-report --run-dir results/kvbench_smollm  --out results/kvbench_smollm/artifacts

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist
