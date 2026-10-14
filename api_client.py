"""
PromptTrace API Client
Centralized client for all backend API interactions with error handling and retries.
Includes response normalization to map backend Pydantic models to UI-friendly structures.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger(__name__)

# Get backend URL from environment or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10  # seconds for general requests
UPLOAD_TIMEOUT = 120  # seconds for file uploads
SIMULATION_TIMEOUT = 900  # seconds for simulations (15 minutes) - prevents timeout on slow GPU runs


class APIError(Exception):
    """Custom exception for API errors."""
    pass


def _safe_get(obj: Any, *keys, default=None) -> Any:
    """Safely navigate nested dict/object structures with fallback."""
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        elif hasattr(current, '__dict__'):
            current = getattr(current, key, default)
        else:
            return default
    return current if current is not None else default


def _ensure_list(val: Any) -> List[Any]:
    """Ensure value is a list, with safe fallback."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _ensure_dict(val: Any) -> Dict[str, Any]:
    """Ensure value is a dict, with safe fallback."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if hasattr(val, '__dict__'):
        return val.__dict__
    return {}


def _extract_text_content(val: Any) -> str:
    """Extract text content from various possible response formats."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        # Try common text fields
        for field in ["summary", "content", "text", "executive_summary", "raw_text", "action_summary"]:
            if field in val and val[field]:
                return str(val[field])
        # Fallback to string representation
        return str(val)
    return str(val)


def normalize_workflow_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize backend workflow response to UI-friendly structure.
    
    Maps backend Pydantic model fields (snake_case) to UI-expected keys
    with defensive fallbacks and type coercion.
    
    Backend fields:
    - retrieved_chunks: List[RAGMetadata]
    - research_output: ResearchAgentOutput
    - analyst_output: AnalystAgentOutput
    - action_output: ActionAgentOutput
    - propagation_history: List[PropagationEvent]
    - individual_agent_times: Dict[str, float]
    - total_processing_time_ms: float
    - workflow_status: str
    - error_message: Optional[str]
    - retrieval_result: Optional[RetrievalResult]
    
    Returns normalized dict with UI-friendly keys:
    - retrieval: {chunks, num_retrieved, retrieval_time_ms, query}
    - research_agent: {summary, key_findings, sources, raw_text, processing_time_ms}
    - analyst_agent: {executive_summary, risk_assessment, detected_instructions, key_recommendations, structured_report, processing_time_ms}
    - action_agent: {action_summary, tool_calls, email_actions, database_actions, file_actions, processing_time_ms}
    - propagation_history: List[event]
    - individual_agent_times: Dict
    - total_processing_time_ms: float
    - status: str
    """
    if not raw_response:
        return {}
    
    # Extract workflow_state from simulation response if wrapped
    workflow_state = raw_response.get("workflow_state", raw_response)
    if not isinstance(workflow_state, dict):
        workflow_state = _ensure_dict(workflow_state)
    
    # --- Normalize Retrieval ---
    retrieved_chunks = _ensure_list(workflow_state.get("retrieved_chunks"))
    retrieval_result = _ensure_dict(workflow_state.get("retrieval_result"))
    
    # Convert RAGMetadata chunks to UI-friendly format
    normalized_chunks = []
    for chunk in retrieved_chunks:
        chunk_dict = _ensure_dict(chunk)
        normalized_chunks.append({
            "chunk_id": chunk_dict.get("chunk_id", ""),
            "document_name": chunk_dict.get("document_name", "Unknown"),
            "page_number": chunk_dict.get("page_number", 0),
            "trust_level": chunk_dict.get("trust_level", "untrusted"),
            "retrieved_text": chunk_dict.get("retrieved_text", ""),
            "chunk_index": chunk_dict.get("chunk_index", 0),
            "score": chunk_dict.get("retrieval_score", 0.0),
            "paragraph_offset": chunk_dict.get("paragraph_offset", 0),
            "character_offset": chunk_dict.get("character_offset", 0),
        })
    
    retrieval = {
        "chunks": normalized_chunks,
        "num_retrieved": len(normalized_chunks),
        "retrieval_time_ms": _safe_get(retrieval_result, "retrieval_time_ms", default=0.0),
        "query": _safe_get(retrieval_result, "query", default=""),
    }
    
    # --- Normalize Research Agent ---
    research_output = _ensure_dict(workflow_state.get("research_output"))
    
    # Handle key_findings which might be a list of strings or list of dicts
    key_findings_raw = _ensure_list(research_output.get("key_findings"))
    key_findings = []
    for kf in key_findings_raw:
        if isinstance(kf, str):
            key_findings.append(kf)
        elif isinstance(kf, dict):
            key_findings.append(_extract_text_content(kf))
        else:
            key_findings.append(str(kf))
    
    # Handle citations/sources
    sources_raw = _ensure_list(research_output.get("sources_cited") or research_output.get("citations"))
    sources = [str(s) for s in sources_raw if s]
    
    # Handle used_chunks
    used_chunks_raw = _ensure_list(research_output.get("used_chunks"))
    used_chunks = []
    for uc in used_chunks_raw:
        uc_dict = _ensure_dict(uc)
        used_chunks.append({
            "chunk_id": uc_dict.get("chunk_id", ""),
            "page_number": uc_dict.get("page_number", 0),
            "document_name": uc_dict.get("document_name", ""),
            "retrieved_text": uc_dict.get("retrieved_text", ""),
        })
    
    research_agent = {
        "summary": _extract_text_content(research_output.get("summary")),
        "key_findings": key_findings,
        "sources": sources,
        "raw_text": _extract_text_content(research_output.get("raw_text")),
        "processing_time_ms": _safe_get(research_output, "processing_time_ms", default=0.0),
        "annotation_summary": _extract_text_content(research_output.get("annotation_summary")),
        "confidence": _safe_get(research_output, "confidence", default=0.0),
        "used_chunks": used_chunks,
    }
    
    # --- Normalize Analyst Agent ---
    analyst_output = _ensure_dict(workflow_state.get("analyst_output"))
    
    # Extract executive summary with fallbacks
    executive_summary = _extract_text_content(
        analyst_output.get("executive_summary") or 
        analyst_output.get("summary")
    )
    
    # Extract risk assessment / threat level
    risk_fields = ["risk_assessment", "threat_level", "confidence_score", "risk_level"]
    risk_assessment = "N/A"
    for rf in risk_fields:
        val = analyst_output.get(rf)
        if val is not None:
            if isinstance(val, (int, float)):
                risk_assessment = f"{val:.0%}" if val <= 1.0 else str(val)
            else:
                risk_assessment = str(val)
            break
    
    # Extract detected instructions
    detected_instructions = ""
    for field in ["detected_instructions", "attack_phrase", "injected_content"]:
        val = analyst_output.get(field)
        if val:
            detected_instructions = str(val)
            break
    
    # Extract key recommendations
    key_recs_raw = _ensure_list(analyst_output.get("key_recommendations") or analyst_output.get("recommendations"))
    key_recommendations = [str(r) for r in key_recs_raw if r]
    
    # Handle structured_report - could be dict or string
    structured_report = analyst_output.get("structured_report", {})
    if isinstance(structured_report, str):
        try:
            import json
            structured_report = json.loads(structured_report)
        except:
            structured_report = {"raw": structured_report}
    
    analyst_agent = {
        "executive_summary": executive_summary,
        "risk_assessment": risk_assessment,
        "detected_instructions": detected_instructions,
        "key_recommendations": key_recommendations,
        "structured_report": _ensure_dict(structured_report),
        "processing_time_ms": _safe_get(analyst_output, "processing_time_ms", default=0.0),
        "annotation_analysis": _extract_text_content(analyst_output.get("annotation_analysis")),
        "confidence": _safe_get(analyst_output, "confidence", default=0.0),
    }
    
    # --- Normalize Action Agent ---
    action_output = _ensure_dict(workflow_state.get("action_output"))
    
    # Extract tool calls
    tool_calls_raw = _ensure_list(action_output.get("tool_calls"))
    tool_calls = []
    for tc in tool_calls_raw:
        tc_dict = _ensure_dict(tc)
        tool_calls.append({
            "type": tc_dict.get("type", ""),
            "parameters": _ensure_dict(tc_dict.get("parameters")),
            "reason": tc_dict.get("reason", ""),
            "operation": tc_dict.get("operation", ""),
            "status": tc_dict.get("status", ""),
        })
    
    # Extract email actions
    email_actions_raw = _ensure_list(action_output.get("email_actions"))
    email_actions = []
    for ea in email_actions_raw:
        ea_dict = _ensure_dict(ea)
        email_actions.append({
            "to": ea_dict.get("to") or ea_dict.get("recipient", "N/A"),
            "subject": ea_dict.get("subject", "N/A"),
            "body": _extract_text_content(ea_dict.get("body"))[:200] if ea_dict.get("body") else "N/A",
            "status": ea_dict.get("status", "simulated"),
        })
    
    # Extract database actions
    db_actions_raw = _ensure_list(action_output.get("database_actions"))
    database_actions = []
    for da in db_actions_raw:
        da_dict = _ensure_dict(da)
        database_actions.append({
            "query": _extract_text_content(da_dict.get("query"))[:300] if da_dict.get("query") else "N/A",
            "query_type": da_dict.get("query_type", "N/A"),
            "database": da_dict.get("database", "N/A"),
            "status": da_dict.get("status", "simulated"),
        })
    
    # Extract file actions
    file_actions_raw = _ensure_list(action_output.get("file_actions"))
    file_actions = []
    for fa in file_actions_raw:
        fa_dict = _ensure_dict(fa)
        file_actions.append({
            "path": fa_dict.get("path", "N/A"),
            "operation": fa_dict.get("operation", "N/A"),
            "status": fa_dict.get("status", "simulated"),
        })
    
    action_agent = {
        "action_summary": _extract_text_content(action_output.get("action_summary")),
        "tool_calls": tool_calls,
        "email_actions": email_actions,
        "database_actions": database_actions,
        "file_actions": file_actions,
        "processing_time_ms": _safe_get(action_output, "processing_time_ms", default=0.0),
    }
    
    # --- Normalize Propagation History ---
    propagation_history_raw = _ensure_list(workflow_state.get("propagation_history"))
    normalized_propagation = []
    for event in propagation_history_raw:
        event_dict = _ensure_dict(event)
        normalized_propagation.append({
            "agent": event_dict.get("agent", "Unknown"),
            "infected": bool(event_dict.get("infected", False)),
            "attack_detected": bool(event_dict.get("attack_detected", False)),
            "attack_phrase": event_dict.get("attack_phrase"),
            "propagation_depth": int(event_dict.get("propagation_depth", 0)),
            "mutation_detected": bool(event_dict.get("mutation_detected", False)),
            "confidence": float(event_dict.get("confidence", 0.0)),
            "category": event_dict.get("category", "UNKNOWN"),
            "timestamp": event_dict.get("timestamp", ""),
        })
    
    # --- Normalize Individual Agent Times ---
    agent_times = _ensure_dict(workflow_state.get("individual_agent_times"))
    normalized_agent_times = {}
    for key, val in agent_times.items():
        try:
            normalized_agent_times[key] = float(val)
        except (ValueError, TypeError):
            normalized_agent_times[key] = 0.0
    
    # --- Build Final Normalized Response ---
    normalized = {
        "simulation_id": raw_response.get("simulation_id", ""),
        "status": raw_response.get("status", workflow_state.get("workflow_status", "unknown")),
        "workflow_state": {
            "retrieval": retrieval,
            "research_agent": research_agent,
            "analyst_agent": analyst_agent,
            "action_agent": action_agent,
            "propagation_history": normalized_propagation,
            "individual_agent_times": normalized_agent_times,
            "total_processing_time_ms": float(_safe_get(workflow_state, "total_processing_time_ms", default=0.0)),
            "error_message": workflow_state.get("error_message"),
        },
        "metrics_snapshot": {
            "total_processing_time_ms": float(_safe_get(workflow_state, "total_processing_time_ms", default=0.0)),
            "agent_times": normalized_agent_times,
            "propagation_events": len(normalized_propagation),
            "infected_agents": [e["agent"] for e in normalized_propagation if e["infected"]],
        },
        "execution_time_ms": float(raw_response.get("execution_time_ms", 0.0)),
    }
    
    return normalized


class APIClient:
    """
    Client for PromptTrace backend API.
    
    Handles:
    - PDF uploads
    - Simulation execution
    - Metrics fetching
    - Logs retrieval
    - Graph data
    """
    
    def __init__(self, backend_url: str = BACKEND_URL):
        """
        Initialize API client.
        
        Args:
            backend_url: Base URL for backend server
        """
        self.backend_url = backend_url.rstrip("/")
        self.session = requests.Session()
        self._verify_backend()
    
    def _verify_backend(self) -> bool:
        """
        Verify backend is accessible.
        
        Returns:
            True if backend is reachable
            
        Raises:
            APIError: If backend is unreachable
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/docs",
                timeout=5,
            )
            if response.status_code in [200, 404]:  # 404 is ok, docs might be disabled
                logger.info(f"Backend verified at {self.backend_url}")
                return True
        except (ConnectionError, Timeout):
            pass
        
        raise APIError(f"Backend unavailable at {self.backend_url}")
    
    def _handle_response(self, response: requests.Response, operation: str) -> Dict[str, Any]:
        """
        Handle API response and raise appropriate errors.
        
        Args:
            response: Response object
            operation: Operation name for logging
            
        Returns:
            Response JSON
            
        Raises:
            APIError: On error responses
        """
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            
            raise APIError(f"{operation} failed: {error_detail}")
        except requests.exceptions.JSONDecodeError:
            raise APIError(f"{operation} returned invalid JSON")
    
    # ======================== PDF Upload ========================
    
    def upload_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Upload PDF to backend and create vector store.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            {
                "status": "success",
                "document_name": str,
                "upload_path": str,
                "chunks_created": int,
                "vector_store_stats": {
                    "total_vectors": int,
                    "embedding_dim": int,
                }
            }
            
        Raises:
            APIError: On upload failure
        """
        if not os.path.exists(file_path):
            raise APIError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith(".pdf"):
            raise APIError("File must be a PDF")
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (Path(file_path).name, f, "application/pdf")}
                response = self.session.post(
                    f"{self.backend_url}/api/upload",
                    files=files,
                    timeout=UPLOAD_TIMEOUT,
                )
            
            result = self._handle_response(response, "PDF upload")
            logger.info(f"PDF uploaded successfully: {file_path}")
            return result
            
        except Timeout:
            raise APIError("Upload timed out - file may be too large")
        except ConnectionError:
            raise APIError(f"Cannot connect to backend at {self.backend_url}")
        except RequestException as e:
            raise APIError(f"Upload failed: {str(e)}")
    
    def list_uploads(self) -> Dict[str, Any]:
        """
        List all uploaded PDFs.
        
        Returns:
            {
                "files": [
                    {
                        "filename": str,
                        "size_mb": float,
                        "path": str,
                    }
                ],
                "count": int,
            }
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/upload/list",
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "List uploads")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch upload list: {str(e)}")
    
    def delete_upload(self, filename: str) -> Dict[str, str]:
        """
        Delete an uploaded PDF.
        
        Args:
            filename: Filename to delete
            
        Returns:
            {"status": "deleted", "filename": str}
            
        Raises:
            APIError: On deletion failure
        """
        try:
            response = self.session.delete(
                f"{self.backend_url}/api/upload/{filename}",
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Delete upload")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot delete upload: {str(e)}")
    
    # ======================== Simulation ========================
    
    def run_simulation(
        self,
        query: str,
        inject_malicious: bool = False,
        malicious_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a simulation with the provided query.
        
        Args:
            query: Query/prompt to analyze
            inject_malicious: Whether to inject malicious content
            malicious_instruction: Custom malicious instruction (optional)
            
        Returns:
            Normalized simulation response with UI-friendly structure.
            
        Raises:
            APIError: On simulation failure
        """
        try:
            payload = {
                "query": query,
                "inject_malicious": inject_malicious,
            }
            if malicious_instruction:
                payload["malicious_instruction"] = malicious_instruction
            
            response = self.session.post(
                f"{self.backend_url}/api/simulate",
                json=payload,
                timeout=SIMULATION_TIMEOUT,
            )
            
            result = self._handle_response(response, "Simulation")
            logger.info(f"Simulation completed: {result.get('simulation_id')}")
            
            # Normalize the response for UI consumption
            normalized = normalize_workflow_response(result)
            logger.debug(f"Normalized workflow state keys: {list(normalized.get('workflow_state', {}).keys())}")
            
            return normalized
            
        except Timeout:
            raise APIError("Simulation timed out - backend may be busy processing")
        except ConnectionError:
            raise APIError(f"Cannot connect to backend at {self.backend_url}")
        except RequestException as e:
            raise APIError(f"Simulation failed: {str(e)}")
    
    def get_simulation_history(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent simulation results.
        
        Args:
            limit: Number of results to return
            
        Returns:
            {
                "simulations": [...],
                "count": int,
            }
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/simulate/history",
                params={"limit": limit},
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Fetch simulation history")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch simulation history: {str(e)}")
    
    # ======================== Metrics ========================
    
    def fetch_metrics(self) -> Dict[str, Any]:
        """
        Fetch current propagation metrics.
        
        Returns:
            {
                "status": "success",
                "metrics": {
                    "attack_success_rate": float,
                    "average_propagation_depth": float,
                    "agent_infection_rate": {...},
                    "mutation_rate": float,
                    "tool_trigger_rate": float,
                    "average_latency_ms": float,
                    "total_runs": int,
                    "successful_detections": int,
                }
            }
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/metrics",
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Fetch metrics")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch metrics: {str(e)}")
    
    def fetch_metrics_summary(self) -> Dict[str, Any]:
        """
        Fetch summary statistics.
        
        Returns:
            {
                "status": "success",
                "summary": {...}
            }
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/metrics/summary",
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Fetch metrics summary")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch metrics summary: {str(e)}")
    
    def fetch_attack_success_rate(self) -> Dict[str, Any]:
        """
        Fetch attack success rate.
        
        Returns:
            {"rate": float, "total_runs": int}
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/metrics/attack-success-rate",
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Fetch attack success rate")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch attack success rate: {str(e)}")
    
    def fetch_propagation_depth(self) -> Dict[str, Any]:
        """
        Fetch average propagation depth.
        
        Returns:
            {"depth": float}
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/metrics/propagation-depth",
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Fetch propagation depth")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch propagation depth: {str(e)}")
    
    # ======================== Logs ========================
    
    def fetch_logs(self, limit: int = 100) -> Dict[str, Any]:
        """
        Fetch propagation event logs.
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            {
                "events": [
                    {
                        "timestamp": str,
                        "agent": str,
                        "infected": bool,
                        "attack_detected": bool,
                        "attack_phrase": str|null,
                        "propagation_depth": int,
                        "mutation_detected": bool,
                        "confidence": float,
                    }
                ],
                "count": int,
            }
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/logs/propagation",
                params={"limit": limit},
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Fetch logs")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch logs: {str(e)}")
    
    def fetch_logs_summary(self) -> Dict[str, Any]:
        """
        Fetch summary of propagation events.
        
        Returns:
            {
                "total_events": int,
                "infected_events": int,
                "agents_affected": [str],
                "average_confidence": float,
            }
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/logs/propagation/summary",
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Fetch logs summary")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch logs summary: {str(e)}")
    
    # ======================== Graph ========================
    
    def fetch_graph(self) -> Dict[str, Any]:
        """
        Fetch propagation graph structure.
        
        Returns:
            {
                "nodes": [
                    {
                        "id": str,
                        "label": str,
                        "infected": bool,
                        "processing_time_ms": float,
                    }
                ],
                "edges": [
                    {
                        "source": str,
                        "target": str,
                        "malicious": bool,
                    }
                ]
            }
            
        Raises:
            APIError: On request failure
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/api/graph",
                timeout=REQUEST_TIMEOUT,
            )
            return self._handle_response(response, "Fetch graph")
        except (Timeout, ConnectionError) as e:
            raise APIError(f"Cannot fetch graph: {str(e)}")
    
    def close(self) -> None:
        """Close the session."""
        self.session.close()


# Singleton instance for use across the app
_client_instance: Optional[APIClient] = None


def get_api_client(backend_url: str = BACKEND_URL) -> APIClient:
    """
    Get or create the API client singleton.
    
    Args:
        backend_url: Backend URL (used on first initialization)
        
    Returns:
        APIClient instance
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = APIClient(backend_url)
    return _client_instance


def reset_api_client() -> None:
    """Reset the API client singleton (useful for testing)."""
    global _client_instance
    if _client_instance:
        _client_instance.close()
    _client_instance = None