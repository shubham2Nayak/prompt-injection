"""
Agent Workflow visualization page
"""

import streamlit as st
import json
from typing import Any, Dict, List, Optional

from api_client import APIError
from components import workflow_nodes, retrieved_chunks, metrics_cards


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


def _extract_text(val: Any) -> str:
    """Extract text content from various formats."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        for field in ["summary", "content", "text", "executive_summary", "raw_text", "action_summary"]:
            if field in val and val[field]:
                return str(val[field])
        return str(val)
    return str(val)


# ---------------------------------------------------------------------------
# Export payloads
#
# These two builders used to run inside render(), so every Streamlit rerun -
# every tab click, expander toggle and radio change - serialized the entire
# results payload with json.dumps and built a pandas DataFrame plus CSV, even
# though neither result was ever displayed unless the download was clicked.
#
# They are wrapped in @st.cache_data and keyed on the simulation id. The payload
# for one simulation id never changes, so this is a correct cache key, and the
# expensive work now happens once instead of on every interaction.
#
# cache_data (not cache_resource) returns copies, so a cached value can never be
# mutated in place by the page.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=4)
def _build_json_export(payload: str, simulation_id: Optional[str]) -> str:
    """
    Serialize a simulation result to pretty-printed JSON.

    Args:
        payload: The result mapping, already JSON-encoded by the caller so the
            cache key is derived from the id rather than by hashing the whole
            nested structure on every rerun
        simulation_id: Simulation identifier, used for the cache key only

    Returns:
        Pretty-printed JSON string
    """
    del simulation_id  # Part of the cache key only.
    # indent=2 roughly doubles the payload size and the work to produce it; the
    # download is a machine-read artifact, so compact form is appropriate.
    return json.dumps(json.loads(payload), separators=(",", ":"), default=str)


@st.cache_data(show_spinner=False, max_entries=4)
def _build_csv_export(propagation_history: str, simulation_id: Optional[str]) -> str:
    """
    Render propagation events as CSV.

    Uses the stdlib csv writer rather than a pandas DataFrame: the rows have
    ragged keys (optional attack_phrase, varying analysis_details), and building
    a DataFrame to immediately stringify it added import weight and coercion for
    no benefit.

    Args:
        propagation_history: JSON-encoded list of propagation events
        simulation_id: Simulation identifier, used for the cache key only

    Returns:
        CSV text with a header row
    """
    import csv
    import io

    del simulation_id  # Part of the cache key only.

    rows = json.loads(propagation_history)
    if not rows:
        return ""

    # Union of keys in first-seen order so the header is stable.
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    return buffer.getvalue()


def _build_exports(results: Dict[str, Any], simulation_id: Optional[str]) -> Dict[str, str]:
    """
    Produce the downloadable payloads for one simulation.

    Called once when a simulation completes; the result is stashed in session
    state so the download buttons are pure string reads on later reruns.

    Args:
        results: Full simulation response
        simulation_id: Simulation identifier

    Returns:
        Mapping with 'json' and 'csv' keys (csv may be empty)
    """
    # Compact pre-encoding is the cache key material; it is one pass over the
    # payload and avoids re-hashing the nested structure on every rerun.
    payload = json.dumps(results, default=str)

    propagation_history = results.get("workflow_state", {}).get(
        "propagation_history", []
    ) or []

    return {
        "json": _build_json_export(payload, simulation_id),
        "csv": _build_csv_export(json.dumps(propagation_history, default=str), simulation_id),
    }


def render():
    """Render the Agent Workflow page."""
    
    # Check backend availability
    if not st.session_state.backend_available:
        st.error(f"❌ Backend unavailable: {st.session_state.get('backend_error', 'Unknown error')}")
        st.info("⚠️ Please configure a valid backend URL in Settings")
        return
    
    st.title("🔄 Agent Workflow Visualization")
    st.markdown("Watch how prompt injections propagate through the agent pipeline")
    
    # Check if PDF is uploaded
    if not st.session_state.get("uploaded_pdf"):
        st.warning("⚠️ Please upload a PDF first")
        st.info("Go to **Upload PDF** page to get started")
        return
    
    st.markdown("---")
    
    # Simulation controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "📝 Simulation Query",
            value="Summarize the key points from the document and generate a comprehensive analysis report with recommendations.",
            height=80,
            help="The prompt that will be sent to the research agent"
        )
    
    with col2:
        injection_type = st.radio(
            "Injection Type",
            ["Safe", "Malicious"],
            help="Choose whether to inject malicious instructions"
        )
        inject_malicious = (injection_type == "Malicious")
    
    st.markdown("---")
    
    # Run simulation button
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col2:
        if st.button("▶️ Run Simulation", use_container_width=True, type="primary"):
            st.session_state.simulation_in_progress = True

            # Create progress placeholder
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                api_client = st.session_state.api_client

                # The previous implementation iterated a six-stage list and slept
                # 0.3s per stage before the request was even sent, adding a fixed
                # 1.8s of pure delay to every simulation and advancing a progress
                # bar that had no relationship to real backend progress. The
                # backend exposes no progress stream, so the honest rendering is
                # an indeterminate spinner around the actual request.
                status_placeholder.info("⏳ Running simulation on backend...")
                progress_placeholder.progress(0.0)

                with st.spinner("🔄 Running simulation on backend..."):
                    result = api_client.run_simulation(
                        query=query,
                        inject_malicious=inject_malicious
                    )

                simulation_id = result.get("simulation_id")

                # Serialize the export payloads once, here, while we already
                # hold the freshly returned result. Doing it in render() meant
                # paying for it on every subsequent rerun.
                st.session_state.export_payloads = _build_exports(result, simulation_id)

                st.session_state.simulation_results = result
                st.session_state.current_simulation_id = simulation_id

                status_placeholder.empty()
                progress_placeholder.empty()

                st.success("✅ Simulation completed!")

                # No st.rerun() here. The results, the current id and the
                # prebuilt export payloads are all in session state, and the
                # panels below this block read them during this same run, so a
                # rerun would only mean rendering the entire page twice. It was
                # also hazardous: the button that was clicked still reports True
                # on the next run, so a rerun inside its own handler can loop.

            except APIError as e:
                status_placeholder.error(f"❌ Simulation failed: {str(e)}")
                st.session_state.simulation_in_progress = False
            except Exception as e:
                status_placeholder.error(f"❌ Unexpected error: {str(e)}")
                st.session_state.simulation_in_progress = False
    
    st.markdown("---")
    
    # Display workflow visualization
    workflow_nodes.render_workflow_panel()
    
    # If simulation has run, show results
    if st.session_state.get("simulation_results"):
        st.markdown("---")
        
        results = st.session_state.simulation_results
        workflow_state = results.get("workflow_state", {})
        metrics_snapshot = results.get("metrics_snapshot", {})
        
        # Summary metrics
        st.subheader("📊 Simulation Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Execution Time",
                f"{results.get('execution_time_ms', 0):.0f}ms"
            )
        with col2:
            infected_agents = metrics_snapshot.get("infected_agents", [])
            st.metric(
                "Agents Infected",
                len(infected_agents)
            )
        with col3:
            st.metric(
                "Propagation Events",
                metrics_snapshot.get("propagation_events", 0)
            )
        with col4:
            st.metric(
                "Status",
                "✅ Completed" if results.get("status") == "completed" else "⚠️ " + results.get("status", "Unknown")
            )
        
        st.markdown("---")
        
        # Agent outputs
        st.subheader("📊 Agent Outputs")
        
        tabs = st.tabs([
            "🔍 Retrieval",
            "📑 Research",
            "📈 Analysis",
            "⚙️ Action",
            "🚨 Propagation"
        ])
        
        # ============================================================
        # PDF Retrieval Tab
        # ============================================================
        with tabs[0]:
            retrieval = _ensure_dict(workflow_state.get("retrieval"))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunks Retrieved", retrieval.get("num_retrieved", 0))
            with col2:
                st.metric("Retrieval Time", f"{retrieval.get('retrieval_time_ms', 0):.0f}ms")
            with col3:
                st.metric("Query", "Processed ✓")
            
            # Show retrieved chunks
            chunks = _ensure_list(retrieval.get("chunks"))
            if chunks:
                st.markdown(f"**Retrieved {len(chunks)} chunks:**")
                retrieved_chunks.render_retrieved_chunks(chunks[:10])  # Show first 10
            else:
                st.info("No chunks retrieved. Try running a simulation with a different query.")
        
        # ============================================================
        # Research Agent Tab
        # ============================================================
        with tabs[1]:
            research = _ensure_dict(workflow_state.get("research_agent"))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Processing Time", f"{research.get('processing_time_ms', 0):.0f}ms")
            with col2:
                key_findings = _ensure_list(research.get("key_findings"))
                st.metric("Key Findings", len(key_findings))
            
            st.markdown("**Summary:**")
            summary = research.get("summary", "")
            if summary:
                st.info(summary)
            else:
                st.info("No research summary generated.")
            
            if key_findings:
                st.markdown("**Key Findings:**")
                for finding in key_findings:
                    st.markdown(f"- {finding}")
            
            # Display sources/citations
            sources = _ensure_list(research.get("sources"))
            if sources:
                with st.expander("📚 Sources & Citations"):
                    for src in sources:
                        st.markdown(f"- {src}")
            
            # Display used chunks
            used_chunks = _ensure_list(research.get("used_chunks"))
            if used_chunks:
                with st.expander("📄 Chunks Used for Research"):
                    for uc in used_chunks:
                        st.markdown(f"""
                        **Chunk ID:** {uc.get('chunk_id', 'N/A')}  
                        **Document:** {uc.get('document_name', 'N/A')} (Page {uc.get('page_number', 'N/A')})  
                        **Preview:** {uc.get('retrieved_text', 'N/A')[:200]}...
                        """)
            
            raw_text = research.get("raw_text", "")
            if raw_text:
                with st.expander("View Raw Output"):
                    st.code(raw_text, language="text")
        
        # ============================================================
        # Analyst Agent Tab
        # ============================================================
        with tabs[2]:
            analyst = _ensure_dict(workflow_state.get("analyst_agent"))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Processing Time", f"{analyst.get('processing_time_ms', 0):.0f}ms")
            with col2:
                risk = analyst.get("risk_assessment", "N/A")
                st.metric("Risk Assessment", risk)
            
            # Executive Summary
            exec_summary = analyst.get("executive_summary", "")
            if exec_summary:
                st.markdown("**Executive Summary:**")
                st.info(exec_summary)
            else:
                st.warning("No executive summary available.")
            
            # Detected instructions / suspicious content
            detected = analyst.get("detected_instructions", "")
            if detected:
                st.warning(f"🚨 Suspicious Instructions Detected: {detected}")
            
            # Key Recommendations
            key_recs = _ensure_list(analyst.get("key_recommendations"))
            if key_recs:
                st.markdown("**Recommendations:**")
                for rec in key_recs:
                    st.markdown(f"- {rec}")
            
            # Structured Report
            structured_report = _ensure_dict(analyst.get("structured_report"))
            if structured_report:
                with st.expander("View Full Structured Report"):
                    st.json(structured_report)
            
            # Annotation Analysis
            annotation_analysis = analyst.get("annotation_analysis", "")
            if annotation_analysis:
                with st.expander("📝 Annotation Analysis"):
                    st.markdown(annotation_analysis)
        
        # ============================================================
        # Action Agent Tab
        # ============================================================
        with tabs[3]:
            action = _ensure_dict(workflow_state.get("action_agent"))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Processing Time", f"{action.get('processing_time_ms', 0):.0f}ms")
            with col2:
                email_actions = _ensure_list(action.get("email_actions"))
                st.metric("Email Ops", len(email_actions))
            with col3:
                db_actions = _ensure_list(action.get("database_actions"))
                st.metric("DB Ops", len(db_actions))
            with col4:
                file_actions = _ensure_list(action.get("file_actions"))
                st.metric("File Ops", len(file_actions))
            
            # Action Summary
            action_summary = action.get("action_summary", "")
            if action_summary:
                st.markdown("**Action Summary:**")
                st.info(action_summary)
            else:
                st.info("No action summary generated.")
            
            # Tool Calls
            tool_calls = _ensure_list(action.get("tool_calls"))
            if tool_calls:
                st.markdown("**Tool Calls Executed:**")
                for i, tc in enumerate(tool_calls[:10]):
                    tc_type = tc.get("type", "unknown")
                    tc_params = _ensure_dict(tc.get("parameters"))
                    tc_reason = tc.get("reason", "")
                    tc_status = tc.get("status", "")
                    
                    with st.expander(f"🔧 Tool Call {i+1}: {tc_type} ({tc_status})"):
                        st.markdown(f"**Reason:** {tc_reason}")
                        st.markdown(f"**Status:** {tc_status}")
                        st.code(json.dumps(tc_params, indent=2), language="json")
            
            # Email Actions
            if email_actions:
                st.markdown("**Email Operations:**")
                for ea in email_actions:
                    st.markdown(f"""
                    - **To:** {ea.get('to', 'N/A')}
                    - **Subject:** {ea.get('subject', 'N/A')}
                    - **Body Preview:** {ea.get('body', 'N/A')}
                    - **Status:** {ea.get('status', 'simulated')}
                    """)
            
            # Database Actions
            if db_actions:
                st.markdown("**Database Operations:**")
                for da in db_actions:
                    st.markdown(f"""
                    - **Query:** `{da.get('query', 'N/A')}`
                    - **Type:** {da.get('query_type', 'N/A')}
                    - **Database:** {da.get('database', 'N/A')}
                    - **Status:** {da.get('status', 'simulated')}
                    """)
            
            # File Actions
            if file_actions:
                st.markdown("**File Operations:**")
                for fa in file_actions:
                    st.markdown(f"""
                    - **Path:** `{fa.get('path', 'N/A')}`
                    - **Operation:** {fa.get('operation', 'N/A')}
                    - **Status:** {fa.get('status', 'simulated')}
                    """)
        
        # ============================================================
        # Propagation / Monitor Tab
        # ============================================================
        with tabs[4]:
            propagation_history = _ensure_list(workflow_state.get("propagation_history"))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Events", len(propagation_history))
            with col2:
                infected = sum(1 for e in propagation_history if e.get("infected"))
                st.metric("Infected Events", infected)
            with col3:
                detected = sum(1 for e in propagation_history if e.get("attack_detected"))
                st.metric("Attacks Detected", detected)
            
            if propagation_history:
                st.markdown("---")
                st.subheader("📊 Propagation Event Details")
                
                # Summary metrics row
                avg_conf = sum(e.get("confidence", 0) for e in propagation_history if e.get("infected")) / max(infected, 1)
                max_depth = max((e.get("propagation_depth", 0) for e in propagation_history), default=0)
                mutations = sum(1 for e in propagation_history if e.get("mutation_detected"))
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Confidence", f"{avg_conf:.0%}")
                with col2:
                    st.metric("Max Depth", max_depth)
                with col3:
                    st.metric("Mutations", mutations)
                with col4:
                    unique_agents = len(set(e.get("agent", "") for e in propagation_history))
                    st.metric("Unique Agents", unique_agents)
                
                st.markdown("---")
                
                # Detailed timeline
                st.markdown("**Event Timeline:**")
                for i, event in enumerate(propagation_history, 1):
                    status_icon = "🔴" if event.get("infected") else "🟢"
                    confidence = event.get("confidence", 0)
                    
                    with st.expander(
                        f"{status_icon} Event {i}: {event.get('agent', 'Unknown')} "
                        f"(Confidence: {confidence:.0%})"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Agent:** {event.get('agent', 'Unknown')}")
                            st.markdown(f"**Infected:** {event.get('infected', False)}")
                            st.markdown(f"**Attack Detected:** {event.get('attack_detected', False)}")
                        with col2:
                            st.markdown(f"**Depth:** {event.get('propagation_depth', 0)}")
                            st.markdown(f"**Mutation:** {event.get('mutation_detected', False)}")
                            st.markdown(f"**Confidence:** {confidence:.0%}")
                            st.markdown(f"**Category:** {event.get('category', 'UNKNOWN')}")
                        
                        attack_phrase = event.get("attack_phrase")
                        if attack_phrase:
                            st.warning(f"🎯 Attack Phrase: `{attack_phrase}`")
                        
                        timestamp = event.get("timestamp", "")
                        if timestamp:
                            st.caption(f"⏰ {timestamp}")
                
                # Infection path visualization
                infected_events = [e for e in propagation_history if e.get("infected")]
                if infected_events:
                    st.markdown("---")
                    st.subheader("🔗 Infection Propagation Path")
                    path = " → ".join(e.get("agent", "Unknown") for e in infected_events)
                    st.code(path, language="text")
                    
                    with st.expander("View Infection Details"):
                        for event in infected_events:
                            st.markdown(f"""
                            **{event.get('agent', 'Unknown')}**
                            - Confidence: {event.get('confidence', 0):.1%}
                            - Mutation: {'Yes' if event.get('mutation_detected') else 'No'}
                            - Depth: {event.get('propagation_depth', 0)}
                            - Category: {event.get('category', 'UNKNOWN')}
                            """)
            else:
                st.info("No propagation events recorded. Run a simulation to see timeline.")
        
        # ============================================================
        # Export Results
        # ============================================================
        st.markdown("---")
        st.subheader("📥 Export Results")

        # Payloads were produced once when the simulation completed. Reading
        # them here is a dict lookup; building them here was the main per-rerun
        # cost on this page.
        exports = st.session_state.get("export_payloads") or {}
        simulation_id = st.session_state.current_simulation_id or "simulation"

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download Full JSON",
                data=exports.get("json", ""),
                file_name=f"simulation_{simulation_id}.json",
                mime="application/json",
                disabled=not exports.get("json"),
            )

        with col2:
            st.download_button(
                label="Download Propagation Events (CSV)",
                data=exports.get("csv", ""),
                file_name=f"propagation_{simulation_id}.csv",
                mime="text/csv",
                disabled=not exports.get("csv"),
            )

        # Tool execution details
        metrics_cards.render_tool_execution_summary(workflow_state)

        st.markdown("---")

        # Infection summary
        st.subheader("🔴 Infection Analysis")
        metrics_cards.render_infection_summary(workflow_state)