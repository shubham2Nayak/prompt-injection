"""
Propagation Timeline page
"""

import streamlit as st
from components import timeline, propagation_graph, logs_panel
from typing import Optional, Dict, Any


def render():
    """Render the Propagation Timeline page."""
    
    st.title("📊 Propagation Timeline")
    st.markdown("Chronological view of attack propagation through agent pipeline")
    
    # Check if simulation has run
    if not st.session_state.get("simulation_results"):
        st.warning("⚠️ No simulation results yet")
        st.info("Go to **Agent Workflow** page to run a simulation first")
        return
    
    results = st.session_state.simulation_results
    workflow_state = results.get("workflow_state", {})
    propagation_events = workflow_state.get("propagation_history", [])
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📋 Timeline Events",
        "🔗 Propagation Graph",
        "📝 System Logs"
    ])
    
    # Timeline view
    with tab1:
        if propagation_events:
            timeline.render_timeline_event(propagation_events[0], 0)
            st.markdown("---")
        
        timeline.render_propagation_timeline(propagation_events)
        
        st.markdown("---")
        
        if propagation_events:
            # Propagation path summary
            st.subheader("Infection Path")
            timeline.render_propagation_path(propagation_events)
    
    # Graph view
    with tab2:
        if propagation_events:
            propagation_graph.render_propagation_graph(propagation_events)
        else:
            st.info("No propagation events to visualize")
    
    # Logs view
    with tab3:
        # Generate mock logs from simulation
        logs = generate_simulation_logs(workflow_state)
        logs_panel.render_logs_panel(logs)


def generate_simulation_logs(workflow_state: Dict[str, Any]) -> list:
    """
    Generate logs from workflow state.
    
    Args:
        workflow_state: Workflow state dictionary
        
    Returns:
        List of log entries
    """
    logs = []
    
    # Parse logs from simulation
    # This is a mock implementation - in production, fetch from backend
    
    logs.append({
        "level": "INFO",
        "timestamp": "2024-01-15 10:00:00",
        "message": "Simulation started"
    })
    
    logs.append({
        "level": "INFO",
        "timestamp": "2024-01-15 10:00:01",
        "message": "PDF retrieval initiated"
    })
    
    retrieval = workflow_state.get("retrieval", {})
    if retrieval.get("num_retrieved", 0) > 0:
        logs.append({
            "level": "INFO",
            "timestamp": "2024-01-15 10:00:02",
            "message": f"Retrieved {retrieval.get('num_retrieved', 0)} document chunks"
        })
    
    logs.append({
        "level": "INFO",
        "timestamp": "2024-01-15 10:00:03",
        "message": "Research agent processing query"
    })
    
    propagation_history = workflow_state.get("propagation_history", [])
    for i, event in enumerate(propagation_history):
        if event.get("infected"):
            logs.append({
                "level": "WARNING",
                "timestamp": f"2024-01-15 10:00:0{4 + i}",
                "message": f"Infection detected in {event.get('agent', 'Unknown')}: {event.get('attack_phrase', 'N/A')[:50]}"
            })
        else:
            logs.append({
                "level": "INFO",
                "timestamp": f"2024-01-15 10:00:0{4 + i}",
                "message": f"{event.get('agent', 'Unknown')} analysis complete - clean"
            })
    
    logs.append({
        "level": "INFO",
        "timestamp": "2024-01-15 10:00:10",
        "message": "Simulation completed successfully"
    })
    
    return logs
