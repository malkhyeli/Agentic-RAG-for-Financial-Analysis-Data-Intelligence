PYTHON ?= ./.venv/bin/python

.PHONY: run test eval smoke-docqa smoke-deepresearch smoke-all check-env

run:
	$(PYTHON) -m streamlit run app_deep_seek.py

test:
	$(PYTHON) -m pytest -q

eval:
	$(PYTHON) -m eval.run --pdf $${PDF:-knowledge/dspy.pdf} --dataset $${DATASET:-eval/dataset.example.json}

# Smoke test: Strict RAG mode (PDF-only, fail-closed)
smoke-docqa:
	$(PYTHON) scripts/smoke_docqa.py --pdf $${PDF:-knowledge/AbuDhabi_ClimateChange_Essay.pdf} --query "$${QUERY:-sea-level rise}"

# Smoke test: DeepResearch mode (web via MCP)
smoke-deepresearch:
	$(PYTHON) scripts/test_deep_research_cli.py --query "$${QUERY:-What are the latest IPCC findings on sea level rise? Provide sources.}"

# Smoke test: Both modes
smoke-all:
	$(PYTHON) scripts/smoke_test_all.py --pdf $${PDF:-knowledge/AbuDhabi_ClimateChange_Essay.pdf} --mode all

# Verify environment is configured (loads .env automatically via Python)
check-env:
	@$(PYTHON) -c "\
import sys; \
from pathlib import Path; \
from dotenv import load_dotenv; \
import os; \
load_dotenv(Path('$(CURDIR)/.env')); \
print('Checking environment variables...'); \
gx = os.getenv('GROUNDX_API_KEY', ''); \
lk = os.getenv('LINKUP_API_KEY', ''); \
sp = os.getenv('SERPER_API_KEY', ''); \
print(f'  GROUNDX_API_KEY: {\"set\" if gx else \"MISSING (required for Strict mode)\"}'); \
print(f'  LINKUP_API_KEY: {\"set\" if lk else \"MISSING\"}'); \
print(f'  SERPER_API_KEY: {\"set\" if sp else \"MISSING\"}'); \
print(f'  DeepResearch: {\"OK (at least one search key)\" if (lk or sp) else \"MISSING (need LINKUP or SERPER key)\"}'); \
"
	@echo "Checking Ollama..."
	@curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && echo "  Ollama: running" || echo "  Ollama: NOT RUNNING (start with: ollama serve)"
	@echo "Checking MCP server..."
	@test -f ../Multi-Agent-deep-researcher-mcp-windows-linux/server.py && echo "  MCP server: found" || echo "  MCP server: NOT FOUND"
