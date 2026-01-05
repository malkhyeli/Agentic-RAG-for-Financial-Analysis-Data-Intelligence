from __future__ import annotations

from typing import Literal, Optional

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from src.agentic_rag.tools.custom_tool import DocumentSearchTool


# Mode type for tool isolation
Mode = Literal["strict", "deepresearch"]


@CrewBase
class AgenticRag:
    """AgenticRag crew with mode-based tool isolation.

    - strict: Uses ONLY DocumentSearchTool (PDF search, no web)
    - deepresearch: Uses ONLY SerperDevTool (web search, no PDF)
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(
        self,
        mode: Mode = "strict",
        pdf_path: Optional[str] = None,
        pdf_tool: Optional[DocumentSearchTool] = None,
    ) -> None:
        """Initialize AgenticRag with mode-based tool isolation.

        Args:
            mode: "strict" for PDF-only RAG, "deepresearch" for web-only research.
            pdf_path: Path to PDF file (required for strict mode if pdf_tool not provided).
            pdf_tool: Pre-initialized DocumentSearchTool (optional, for strict mode).

        Raises:
            ValueError: If strict mode is selected but no PDF path or tool is provided.
        """
        self._mode = mode
        self._pdf_path = pdf_path
        self._pdf_tool = pdf_tool
        self._web_tool: Optional[SerperDevTool] = None

        # Validate and initialize tools based on mode
        if mode == "strict":
            if pdf_tool is not None:
                self._pdf_tool = pdf_tool
            elif pdf_path:
                self._pdf_tool = DocumentSearchTool(file_path=pdf_path)
            else:
                raise ValueError(
                    "Strict mode requires either pdf_path or pdf_tool. "
                    "Provide a PDF path or pre-initialized DocumentSearchTool."
                )
            # Strict mode: NO web tool
            self._web_tool = None
        elif mode == "deepresearch":
            # DeepResearch mode: NO PDF tool, ONLY web tool
            self._pdf_tool = None
            self._web_tool = SerperDevTool()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'strict' or 'deepresearch'.")

    @property
    def mode(self) -> Mode:
        """Current operating mode."""
        return self._mode

    @property
    def tools(self) -> list:
        """Return tools based on current mode (enforces isolation)."""
        if self._mode == "strict":
            assert self._pdf_tool is not None, "Strict mode requires pdf_tool"
            assert self._web_tool is None, "Strict mode must not have web_tool"
            return [self._pdf_tool]
        elif self._mode == "deepresearch":
            assert self._pdf_tool is None, "DeepResearch mode must not have pdf_tool"
            assert self._web_tool is not None, "DeepResearch mode requires web_tool"
            return [self._web_tool]
        return []

    @agent
    def retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['retriever_agent'],
            verbose=True,
            tools=self.tools,  # Mode-isolated tools
        )

    @agent
    def response_synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['response_synthesizer_agent'],
            verbose=True
        )

    @task
    def retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieval_task'],
        )

    @task
    def response_task(self) -> Task:
        return Task(
            config=self.tasks_config['response_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AgenticRag crew with mode-isolated tools."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


def build_crew(
    mode: Mode,
    *,
    pdf_path: Optional[str] = None,
    pdf_tool: Optional[DocumentSearchTool] = None,
) -> AgenticRag:
    """Factory function to build a crew with mode-based tool isolation.

    Args:
        mode: "strict" for PDF-only, "deepresearch" for web-only.
        pdf_path: Path to PDF (required for strict mode if pdf_tool not provided).
        pdf_tool: Pre-initialized DocumentSearchTool (optional).

    Returns:
        Configured AgenticRag instance.

    Raises:
        ValueError: If strict mode is selected but no PDF is provided.
    """
    return AgenticRag(mode=mode, pdf_path=pdf_path, pdf_tool=pdf_tool)


def validate_tool_isolation(rag: AgenticRag) -> None:
    """Runtime assertion to validate tool isolation is correct.

    Raises:
        AssertionError: If tool isolation is violated.
    """
    tools = rag.tools
    tool_names = {type(t).__name__ for t in tools}

    if rag.mode == "strict":
        assert "DocumentSearchTool" in tool_names, "Strict mode must have DocumentSearchTool"
        assert "SerperDevTool" not in tool_names, "Strict mode must NOT have SerperDevTool"
        assert len(tools) == 1, f"Strict mode should have exactly 1 tool, got {len(tools)}"
    elif rag.mode == "deepresearch":
        assert "SerperDevTool" in tool_names, "DeepResearch mode must have SerperDevTool"
        assert "DocumentSearchTool" not in tool_names, "DeepResearch mode must NOT have DocumentSearchTool"
        assert len(tools) == 1, f"DeepResearch mode should have exactly 1 tool, got {len(tools)}"
