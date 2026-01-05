<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">


# AGENTIC-RAG-FOR-FINANCIAL-ANALYSIS-DATA-INTELLIGENCE

<em>Transform Data into Actionable Financial Insights</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/last-commit/malkhyeli/Agentic-RAG-for-Financial-Analysis-Data-Intelligence?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/malkhyeli/Agentic-RAG-for-Financial-Analysis-Data-Intelligence?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/malkhyeli/Agentic-RAG-for-Financial-Analysis-Data-Intelligence?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=FastAPI&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/Pytest-0A9EDC.svg?style=flat&logo=Pytest&logoColor=white" alt="Pytest">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat&logo=YAML&logoColor=white" alt="YAML">

</div>
<br>

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)

---

## Overview

Agentic-RAG-for-Financial-Analysis-Data-Intelligence is a comprehensive framework that combines retrieval-augmented generation with multi-agent orchestration to deliver transparent and reliable financial insights. It emphasizes strict evidence sourcing from PDFs and web sources, ensuring responses are grounded in verified data.

**Why Agentic-RAG?**

This project streamlines complex data workflows for financial analysis. The core features include:

- 🧩 **🔍 Document Grounding:** Ensures answers are based solely on uploaded PDFs or web sources, maintaining high accuracy and transparency.
- 🧠 **🤖 Multi-Agent Orchestration:** Coordinates autonomous agents for scalable, flexible research and response generation.
- 🛠️ **📝 Environment & Validation Tools:** Facilitates environment setup, dependency management, and resource validation for robust development.
- 📄 **📚 Document Ingestion & Retrieval:** Supports secure PDF uploads, indexing, and precise document search.
- ⚙️ **🔧 Modular Architecture:** Configurable via YAML and scripts, enabling tailored workflows and easy maintenance.
- 🔒 **🚫 Fail-Closed Responses:** Prevents hallucinations by refusing to answer when evidence is insufficient, boosting system trustworthiness.

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Build Agentic-RAG-for-Financial-Analysis-Data-Intelligence from the source and install dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/malkhyeli/Agentic-RAG-for-Financial-Analysis-Data-Intelligence
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd Agentic-RAG-for-Financial-Analysis-Data-Intelligence
    ```

3. **Install the dependencies:**

**Using [pip](https://pypi.org/project/pip/):**

```sh
❯ pip install -r requirements-dev.txt, requirements.txt
```

### Usage

Run the project with:

**Using [pip](https://pypi.org/project/pip/):**

```sh
python {entrypoint}
```

### Testing

Agentic-rag-for-financial-analysis-data-intelligence uses the {__test_framework__} test framework. Run the test suite with:

**Using [pip](https://pypi.org/project/pip/):**

```sh
pytest
```

---

<div align="left"><a href="#top">⬆ Return</a></div>

---


## Key Properties
- **Fail-closed by default**: No evidence → no answer.
- **Agentic separation of concerns**: retrieval and synthesis are independent agents.
- **Grounding enforcement**: synthesis is blocked unless retriever output passes validation.
- **Debuggable**: optional UI debug panel exposes retriever output for inspection.

### API Keys
- GroundX API key: https://docs.eyelevel.ai/documentation/fundamentals/quickstart#step-1-getting-your-api-key

Create a `.env` file (see `.env.example`) or use Streamlit secrets to provide:
- `GROUNDX_API_KEY` (Document QA over PDFs)

**For DeepResearch mode (web search), choose ONE:**
- `SERPER_API_KEY` **(recommended)** - Google Search via Serper (https://serper.dev/)
- `LINKUP_API_KEY` - LinkUp search (https://www.linkup.so/)

Optional configuration:
- `RESEARCH_SEARCH_PROVIDER` - `serper`, `linkup`, or `auto` (default: auto)
- `DOCQA_LLM_MODEL` (e.g. `ollama/deepseek-r1:7b`)
- `OLLAMA_BASE_URL` (e.g. `http://localhost:11434`)

### Watch this tutorial on YouTube
[![Watch this tutorial on YouTube](https://github.com/patchy631/ai-engineering-hub/blob/main/agentic_rag_deepseek/assets/thumbnail.png)](https://www.youtube.com/watch?v=79xvgj4wvHQ)

---
## Two-Mode Architecture

The application supports two distinct modes, each with strict fail-closed behavior:

| Mode | Data Source | Fail-Closed Behavior |
|------|-------------|---------------------|
| **Strict RAG** | Uploaded PDF only | Returns "Not in the provided documents." when evidence is insufficient |
| **DeepResearch** | Web search only | Returns "No web sources retrieved." when no valid citations found |

**Mode isolation is enforced**: Strict mode never accesses web/MCP, DeepResearch never accesses PDF/GroundX.

---
## How to Run (3 Commands)

### 1. Start Ollama (local LLM)
```bash
# Terminal 1: Start Ollama server
ollama serve

# Pull the model (first time only)
ollama pull deepseek-r1:7b
```

### 2. (Optional) Start MCP Server for DeepResearch
```bash
# Terminal 2: Only needed if using DeepResearch mode
cd ../Multi-Agent-deep-researcher-mcp-windows-linux
uv sync  # or: pip install -r requirements.txt
uv run server.py
```
> Note: The MCP server is auto-launched by the app when needed. This step is optional for debugging.

### 3. Start Streamlit App
```bash
# Terminal 3: Run the main application
cd agentic_rag_deepseek
source .venv/bin/activate
make run  # or: python -m streamlit run app_deep_seek.py
```

---
## Setup and Installation

**1. Environment Setup**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

**2. Install Dependencies** (Python 3.11+):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install groundx crewai streamlit litellm httpx pypdf
```

**3. Verify Setup**:
```bash
make check-env
```

---
## Smoke Tests

Verify both modes work correctly:

```bash
# Test Strict RAG mode only
make smoke-docqa PDF=knowledge/AbuDhabi_ClimateChange_Essay.pdf

# Test DeepResearch mode only
make smoke-deepresearch

# Test both modes
make smoke-all PDF=knowledge/AbuDhabi_ClimateChange_Essay.pdf
```

---
## Fail-Closed Behavior (Expected)
- If retriever evidence is missing or invalid, the app responds with:
  **"Not in the provided documents."**
- This is intentional and indicates hallucination prevention.
- If you want web sources instead, switch to **DeepResearch (Web)** mode.

## 📬 Stay Updated with Our Newsletter!
**Get a FREE Data Science eBook** 📖 with 150+ essential lessons in Data Science when you subscribe to our newsletter! Stay in the loop with the latest tutorials, insights, and exclusive resources. [Subscribe now!](https://join.dailydoseofds.com)

[![Daily Dose of Data Science Newsletter](https://github.com/patchy631/ai-engineering/blob/main/resources/join_ddods.png)](https://join.dailydoseofds.com)

---

## Demo Script
1. Ask a question clearly answered by the document.
2. Ask a question not present in the document.
3. Ask a tempting hallucination prompt.

Correct behavior is to answer (1) and refuse (2)–(3).

---

## Verification Checklist

Use this checklist to verify both modes are working correctly:

### Strict RAG Mode
- [ ] Upload a PDF document
- [ ] Ask a question answered by the document → Should return answer with `(p.X)` citations
- [ ] Ask a question NOT in the document → Should return "Not in the provided documents."
- [ ] Answer should NEVER contain URLs, DOIs, or web references
- [ ] Evidence section should show PDF quotes with page numbers

### DeepResearch Mode
- [ ] Switch to DeepResearch mode (no PDF needed)
- [ ] Ask a question requiring web research → Should return answer with Sources URLs
- [ ] All sources should be full URLs (no shorteners like bit.ly)
- [ ] If no sources found → Should return "No web sources retrieved."
- [ ] Should NEVER access the uploaded PDF or GroundX

### Mode Isolation
- [ ] Strict mode should never import `mcp_runner` module
- [ ] DeepResearch mode should never import `DocumentSearchTool`
- [ ] Run `make test` to verify isolation with unit tests

---
## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
