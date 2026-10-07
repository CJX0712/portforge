# PortForge · 世界级投资组合优化系统 — 作者：晨星 (CJX0712)
.PHONY: test lint format demo freeze ci

test:
	python -m pytest -q -W ignore::UserWarning

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

demo:
	python examples/run_demo.py

freeze:
	python -m pip freeze > requirements.lock.txt

ci: lint test demo

publish:
	python ../../.workbuddy/skills/random-ai-system-delivery/scripts/gh_push.py \
		--repo portforge --local . \
		--description "PortForge · 世界级投资组合优化系统（作者：晨星）" \
		--tag v0.1.0
