"""
PromptTrace State Models
Pydantic models for shared workflow state and data structures.
"""

from typing import Any, List, Dict, Optional, Literal, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from context.schema import ContextAnnotation


class RAGMetadata(BaseModel):
    """Metadata for retrieved RAG chunks."""

    document_name: str = Field(..., description="Name of the source PDF document")
    page_number: int = Field(..., description="Page number in the PDF")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    trust_level: Literal["trusted", "untrusted"] = Field(
        "untrusted", description="Trust level of the chunk"
    )
    retrieved_text: str = Field(..., description="The actual text content")
    chunk_index: int = Field(..., description="Sequential index of chunk in document")
    annotation_summary: str = Field(
        "", description="Summary of context annotations attached to this chunk"
    )
    paragraph_offset: int = Field(
        0, description="Paragraph offset within the source document"
    )
    character_offset: int = Field(
        0, description="Character offset within the paragraph"
    )


class RetrievalResult(BaseModel):
    """Result from RAG retrieval."""

    chunks: List[RAGMetadata] = Field(default_factory=list)
    query: str = Field(..., description="Original query used for retrieval")
    num_retrieved: int = Field(..., description="Number of chunks retrieved")
    retrieval_time_ms: float = Field(..., description="Time taken for retrieval")


class ResearchAgentOutput(BaseModel):
    """Output from Research Agent."""

    summary: str = Field(..., description="Faithful summary of retrieved chunks")
    key_findings: List[str] = Field(
        default_factory=list, description="Extracted key points"
    )
    sources_cited: List[str] = Field(default_factory=list, description="Source chunks")
    raw_text: str = Field(
        ..., description="Raw unfiltered text including any injected content"
    )
    processing_time_ms: float = Field(
        ..., description="Agent processing time"
    )
    citations: List[str] = Field(
        default_factory=list, description="Chunk IDs cited in summary"
    )
    used_chunks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chunks used for summary with metadata"
    )
    annotation_summary: str = Field(
        "", description="Summary of chunk annotations"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Annotation confidence score"
    )


class AnalystAgentOutput(BaseModel):
    """Output from Analyst Agent."""

    executive_summary: str = Field(
        ..., description="Executive summary of the analysis"
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings from the analysis"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations from the analysis"
    )
    annotation_analysis: str = Field(
        ..., description="Analysis of how document context was preserved, transformed, or propagated"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Analyst confidence score (0-1)"
    )
    processing_time_ms: float = Field(
        ..., description="Agent processing time"
    )


class ActionAgentOutput(BaseModel):
    """Output from Action Agent."""

    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list, description="Simulated tool calls to execute"
    )
    action_summary: str = Field(..., description="Summary of actions to take")
    email_actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Email operations attempted"
    )
    database_actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Database operations attempted"
    )
    file_actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="File operations attempted"
    )
    processing_time_ms: float = Field(..., description="Agent processing time")


class PropagationEvent(BaseModel):
    """Single propagation detection event."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str = Field(..., description="Agent that was analyzed")
    infected: bool = Field(..., description="Whether infection was detected")
    attack_detected: bool = Field(
        ..., description="Whether an attack was detected in output"
    )
    attack_phrase: Optional[str] = Field(
        None, description="Detected attack phrase or instruction"
    )
    propagation_depth: int = Field(
        ..., description="How many hops from original injection"
    )
    mutation_detected: bool = Field(
        ..., description="Whether the injection mutated"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of detection (0-1)"
    )
    category: str = Field(
        ..., description="Annotation category (e.g., EMBEDDED_PROCESSING_NOTE, ACADEMIC_DESCRIPTION)"
    )


class PropagationMonitorOutput(BaseModel):
    """Output from Propagation Monitor."""

    infected: bool = Field(..., description="Whether propagation detected")
    agent: str = Field(..., description="Agent being monitored")
    attack_detected: bool = Field(..., description="Attack detected in output")
    attack_phrase: Optional[str] = Field(None, description="Detected malicious phrase")
    propagation_depth: int = Field(..., description="Propagation depth from origin")
    mutation_detected: bool = Field(..., description="Mutation detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp",
    )
    analysis_details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed analysis results"
    )


class WorkflowState(BaseModel):
    """Complete shared state for the LangGraph workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input
    original_query: str = Field(..., description="Original user query")
    uploaded_pdf_path: Optional[str] = Field(None, description="Path to uploaded PDF")

    # RAG
    retrieval_result: Optional[RetrievalResult] = Field(
        None, description="Result from RAG retrieval"
    )
    retrieved_chunks: List[RAGMetadata] = Field(
        default_factory=list, description="Retrieved chunks from vector store"
    )
    chunk_annotations: List["ContextAnnotation"] = Field(
        default_factory=list,
        description="Context annotations attached to each retrieved chunk",
    )

    # Agent Outputs
    research_output: Optional[ResearchAgentOutput] = Field(
        None, description="Output from Research Agent"
    )
    analyst_output: Optional[AnalystAgentOutput] = Field(
        None, description="Output from Analyst Agent"
    )
    action_output: Optional[ActionAgentOutput] = Field(
        None, description="Output from Action Agent"
    )

    # Propagation Monitoring
    propagation_history: List[PropagationEvent] = Field(
        default_factory=list, description="History of propagation detections"
    )
    propagation_timeline: List[PropagationMonitorOutput] = Field(
        default_factory=list, description="Chronological propagation timeline"
    )

    # Workflow Metadata
    workflow_status: Literal["started", "in_progress", "completed", "failed"] = Field(
        "started", description="Current workflow status"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    start_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Workflow start time"
    )
    end_timestamp: Optional[datetime] = Field(None, description="Workflow end time")

    # Agent Status
    agents_status: Dict[str, Literal["pending", "running", "completed", "failed"]] = (
        Field(
            default_factory=dict,
            description="Status of each agent",
        )
    )

    # Metrics and Performance
    total_processing_time_ms: float = Field(default=0.0)
    individual_agent_times: Dict[str, float] = Field(default_factory=dict)


class PropagationMetrics(BaseModel):
    """Aggregated propagation metrics."""

    attack_success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of queries with successful attacks"
    )
    average_propagation_depth: float = Field(
        ..., description="Average number of hops before detection"
    )
    propagation_paths: List[List[str]] = Field(
        default_factory=list, description="Paths of propagation through agents"
    )
    agent_infection_rate: Dict[str, float] = Field(
        default_factory=dict, description="Infection rate per agent"
    )
    mutation_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Rate of injection mutation"
    )
    tool_trigger_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Rate of tool execution from attacks"
    )
    average_latency_ms: float = Field(..., description="Average end-to-end latency")
    total_runs: int = Field(..., description="Total number of runs analyzed")
    successful_detections: int = Field(
        ..., description="Number of successfully detected attacks"
    )
    persistence_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of infections that persist across agents"
    )
    annotation_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Average annotation confidence score"
    )
    false_positive_count: int = Field(
        ..., ge=0, description="Count of false positive annotations"
    )
    false_negative_count: int = Field(
        ..., ge=0, description="Count of false negative annotations"
    )


class SimulationRequest(BaseModel):
    """Request to run a simulation."""

    query: str = Field(..., description="Query/prompt to inject")
    pdf_path: Optional[str] = Field(
        None, description="Path to PDF file (if using existing upload)"
    )
    inject_malicious: bool = Field(
        True, description="Whether to inject malicious instructions"
    )
    malicious_instruction: Optional[str] = Field(
        None, description="Custom malicious instruction to inject"
    )


class SimulationResponse(BaseModel):
    """Response from a simulation run."""

    simulation_id: str = Field(..., description="Unique simulation ID")
    status: str = Field(..., description="Simulation status")
    workflow_state: WorkflowState = Field(..., description="Complete workflow state")
    metrics_snapshot: Dict[str, Any] = Field(
        default_factory=dict, description="Metrics at end of simulation"
    )
    execution_time_ms: float = Field(..., description="Total execution time")
