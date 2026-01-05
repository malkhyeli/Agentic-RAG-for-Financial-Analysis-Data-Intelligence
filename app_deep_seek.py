from __future__ import annotations

# =============================================================================
# Financial Audit & Risk/Compliance RAG System
# 100% Document-Grounded - No Web Access
# =============================================================================
import os
from pathlib import Path

# Load .env FIRST (before any library imports)
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_PATH, override=False)

# Disable CrewAI/OpenTelemetry telemetry BEFORE importing crewai
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

# =============================================================================
# Imports - No web/MCP dependencies
# =============================================================================
import base64
import gc
import hashlib
import re
from io import BytesIO
from typing import Optional

import streamlit as st
from crewai import LLM

from src.agentic_rag.tools.custom_tool import DocumentSearchTool, MissingApiKeyError
from src.docqa.pipeline import DocQAConfig, run_docqa, Evidence
from src.utils.pdf_store import StoredPdf, ensure_pdf_in_session


# =============================================================================
# Constants - Fail-Closed Compliance
# =============================================================================
REFUSAL_TEXT = "Not in document."

# Patterns that indicate ACTUAL web URLs - only reject real URLs, not words like "company"
WEB_URL_PATTERN = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+)",
    re.IGNORECASE
)


# =============================================================================
# Streamlit Config
# =============================================================================
st.set_page_config(
    page_title="Financial Analysis & Data Intelligence RAG",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_llm():
    """Load LLM for document QA."""
    return LLM(
        model=os.getenv("DOCQA_LLM_MODEL", "ollama/deepseek-r1:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )


def reset_chat():
    """Clear chat history."""
    st.session_state.messages = []
    gc.collect()


def reset_document():
    """Clear loaded document."""
    st.session_state.pdf_tool = None
    st.session_state.pdf_store = None
    st.session_state.pdf_hash = None


def display_pdf(file_bytes: bytes, file_name: str):
    """Display PDF preview in sidebar."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="600px"
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview: {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)


def _merge_pdfs(pdf_blobs: list[bytes]) -> bytes:
    """Merge multiple PDFs into one."""
    if len(pdf_blobs) == 1:
        return pdf_blobs[0]

    try:
        from pypdf import PdfReader, PdfWriter
        writer = PdfWriter()
        for blob in pdf_blobs:
            reader = PdfReader(BytesIO(blob))
            for page in reader.pages:
                writer.add_page(page)
        out = BytesIO()
        writer.write(out)
        return out.getvalue()
    except Exception:
        pass

    from PyPDF2 import PdfReader, PdfWriter
    writer = PdfWriter()
    for blob in pdf_blobs:
        reader = PdfReader(BytesIO(blob))
        for page in reader.pages:
            writer.add_page(page)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _contains_web_url(text: str) -> bool:
    """Check if text contains actual web URLs (not words like 'company')."""
    return bool(WEB_URL_PATTERN.search(text or ""))


def _filter_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Filter out evidence containing actual web URLs."""
    filtered = []
    for e in evidence:
        if not _contains_web_url(e.quote):
            filtered.append(e)
    return filtered


def format_audit_response(
    query: str,
    answer: str,
    evidence: list[Evidence],
    decision: str,
) -> str:
    """Format response for Financial Audit/Compliance use case.

    Structure:
    - Evidence: Verbatim quotes with (p.#) citations
    - Findings: Minimal paraphrases from evidence only
    - Risk Flags: Only if explicitly in evidence
    """
    # If decision is abstain or no evidence, refuse
    if decision == "abstain" or not evidence:
        return REFUSAL_TEXT

    # Filter out evidence with actual URLs (but keep normal document text)
    clean_evidence = _filter_evidence(evidence)
    if not clean_evidence:
        return REFUSAL_TEXT

    # Check if the original answer was a refusal (exact match only)
    answer_stripped = (answer or "").strip()
    if not answer_stripped or answer_stripped == "Not in the provided documents.":
        return REFUSAL_TEXT

    # Build Evidence section (max 5 quotes)
    evidence_lines = []
    for e in clean_evidence[:5]:
        quote = e.quote.strip()
        if quote:
            evidence_lines.append(f"• \"{quote}\" (p.{e.page})")

    if not evidence_lines:
        return REFUSAL_TEXT

    evidence_section = "## Evidence\n" + "\n".join(evidence_lines)

    # Build Findings section from the LLM answer
    # Strip any markdown headers the LLM might have added
    findings_text = answer_stripped
    findings_text = re.sub(r"^#+\s*Evidence.*$", "", findings_text, flags=re.MULTILINE | re.IGNORECASE)
    findings_text = re.sub(r"^#+\s*Findings.*$", "", findings_text, flags=re.MULTILINE | re.IGNORECASE)
    findings_text = re.sub(r"^#+\s*Risk.*$", "", findings_text, flags=re.MULTILINE | re.IGNORECASE)
    findings_text = findings_text.strip()

    if not findings_text:
        # Use the evidence as the findings if LLM didn't generate text
        findings_text = "See evidence above."

    findings_section = "## Findings\n" + findings_text

    # Build Risk Flags section - only if risk keywords appear in evidence
    risk_keywords = ["risk", "compliance", "violation", "deficiency", "weakness",
                     "material", "audit", "finding", "control", "fraud", "error",
                     "misstatement", "irregularity", "non-compliance"]

    evidence_text_lower = " ".join(e.quote.lower() for e in clean_evidence)
    has_risk_content = any(kw in evidence_text_lower for kw in risk_keywords)

    if has_risk_content:
        risk_flags = []
        for e in clean_evidence:
            quote_lower = e.quote.lower()
            if any(kw in quote_lower for kw in risk_keywords):
                risk_flags.append(f"• See evidence from p.{e.page}")

        if risk_flags:
            risk_section = "\n\n## Risk Flags\n" + "\n".join(risk_flags[:3])
        else:
            risk_section = ""
    else:
        risk_section = ""

    return f"{evidence_section}\n\n{findings_section}{risk_section}"


# =============================================================================
# Session State
# =============================================================================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pdf_tool", None)
st.session_state.setdefault("pdf_store", None)
st.session_state.setdefault("pdf_hash", None)


# =============================================================================
# Sidebar - Document Upload Only (No Mode Selection)
# =============================================================================
with st.sidebar:
    st.header("📄 Document Upload")
    st.caption("Upload audit reports, financial statements, or compliance documents.")

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        blobs = [f.getvalue() for f in uploaded_files]
        merged = _merge_pdfs(blobs)
        digest = hashlib.sha256(merged).hexdigest()[:12]
        filename = f"documents_{digest}.pdf"
        stored = ensure_pdf_in_session(
            st.session_state,
            data=merged,
            filename=filename,
            session_key="pdf_store"
        )
        st.session_state.pdf_store = stored

        # Force reindex if document changed
        if st.session_state.pdf_hash != stored.sha256:
            st.session_state.pdf_hash = stored.sha256
            st.session_state.pdf_tool = None

    stored: Optional[StoredPdf] = st.session_state.pdf_store
    if stored is not None and stored.path.exists():
        st.success(f"Loaded: `{stored.filename}`")

        if st.session_state.pdf_tool is None:
            try:
                with st.spinner("Indexing document..."):
                    st.session_state.pdf_tool = DocumentSearchTool(
                        file_path=str(stored.path)
                    )
                # Check actual readiness status
                tool = st.session_state.pdf_tool
                if getattr(tool, "ready", False):
                    st.success("Document indexed. Ready for queries.")
                elif getattr(tool, "in_progress", False):
                    st.warning("Document still processing. Some queries may fail.")
                else:
                    st.info("Document uploaded. Indexing status unknown.")
            except MissingApiKeyError as e:
                st.error(str(e))
                st.caption("Add `GROUNDX_API_KEY` to `.env`")
            except Exception as e:
                st.error(f"Indexing failed: {type(e).__name__}: {e}")

        # PDF preview
        try:
            display_pdf(stored.path.read_bytes(), stored.filename)
        except Exception:
            pass

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.button("Clear Chat", on_click=reset_chat)
    with col_b:
        if st.button("Clear Doc"):
            reset_document()
            reset_chat()
            st.rerun()

    st.divider()
    st.caption("**Compliance Mode**: Document-grounded only. No web access.")


# =============================================================================
# Main Chat Interface
# =============================================================================
st.markdown(
    """
    # 📊 Financial Analysis & Data Intelligence RAG

    **Document-Grounded Analysis** — All responses are strictly based on uploaded documents.
    """
)

st.info(
    "⚠️ **Fail-Closed System**: If information is not in the document, "
    "the system will respond with \"Not in document.\" No external sources are used."
)

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask about your audit documents...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            # Check for web search requests - refuse immediately
            web_request_patterns = [
                "search the web", "search online", "look up online",
                "google", "find on internet", "web search"
            ]
            if any(p in prompt.lower() for p in web_request_patterns):
                answer = REFUSAL_TEXT
            elif st.session_state.pdf_tool is None:
                answer = "Please upload a PDF document to begin analysis."
            else:
                # Run document QA
                res = run_docqa(
                    prompt,
                    pdf_tool=st.session_state.pdf_tool,
                    llm=load_llm(),
                    config=DocQAConfig()
                )

                # Format for audit/compliance output
                answer = format_audit_response(
                    query=prompt,
                    answer=res.answer or "",
                    evidence=res.evidence,
                    decision=res.decision,
                )

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
