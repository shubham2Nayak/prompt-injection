"""
Timeline component for displaying propagation events
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any


def render_timeline_event(event: Dict[str, Any], index: int):
    """Render a single timeline event card."""
    
    # Color coding based on status
    status_colors = {
        "infected": "#FF3333",
        "detected": "#FFD700",
        "safe": "#00DD00",
    }
    
    status = "infected" if event.get("infected") else "safe"
    color = status_colors.get(status, "#0066FF")
    
    col1, col2 = st.columns([0.5, 4])
    
    with col1:
        # Timeline marker
        st.markdown(f"""
        <div style="
            width: 30px;
            height: 30px;
            background-color: {color};
            border-radius: 50%;
            border: 3px solid white;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            margin-top: 10px;
        ">{index + 1}</div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Event card
        container = st.container()
        
        with container:
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{event.get('agent', 'Unknown')}**")
            with cols[1]:
                if event.get("infected"):
                    st.markdown('<span style="color: #FF3333;">🔴 Infected</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color: #00DD00;">✅ Safe</span>', unsafe_allow_html=True)
            
            # Timestamp
            timestamp = event.get("timestamp", "")
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp_str = str(timestamp)
            st.caption(f"⏰ {timestamp_str}")
            
            # Message
            message = event.get("message", "")
            if message:
                st.markdown(f"{message}")
            
            # Confidence level (for infected events)
            if event.get("infected"):
                confidence = event.get("confidence", 0)
                st.markdown(f"**Infection Confidence:** {confidence:.1%}")
                
                # Attack phrase if detected
                attack_phrase = event.get("attack_phrase")
                if attack_phrase:
                    st.warning(f"**Detected Phrase:** `{attack_phrase}`")
                
                # Propagation depth
                depth = event.get("propagation_depth", 1)
                st.markdown(f"**Propagation Depth:** {depth} hop{'s' if depth > 1 else ''}")


def render_propagation_timeline(events: List[Dict[str, Any]]):
    """
    Render the propagation timeline.
    
    Args:
        events: List of propagation event dictionaries
    """
    if not events:
        st.info("No propagation events recorded. Run a simulation to see timeline.")
        return
    
    st.subheader("📊 Propagation Timeline")
    st.markdown("Chronological view of attack propagation through agent pipeline")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    infected_events = sum(1 for e in events if e.get("infected"))
    total_events = len(events)
    avg_confidence = sum(e.get("confidence", 0) for e in events if e.get("infected")) / max(infected_events, 1)
    max_depth = max((e.get("propagation_depth", 1) for e in events), default=0)
    
    with col1:
        st.metric("Total Events", total_events)
    with col2:
        st.metric("Infected Events", infected_events)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    with col4:
        st.metric("Max Depth", max_depth)
    
    st.markdown("---")
    
    # Timeline visualization
    st.markdown("### Event Timeline")
    
    for i, event in enumerate(events):
        render_timeline_event(event, i)


def render_propagation_path(events: List[Dict[str, Any]]):
    """Render a visual propagation path."""
    
    if not events:
        return
    
    infected_agents = [e["agent"] for e in events if e.get("infected")]
    
    if not infected_agents:
        st.success("✅ No infection detected in pipeline")
        return
    
    st.warning("🔴 Infection Propagation Path:")
    
    # Create path visualization
    path = " → ".join(infected_agents)
    st.code(path, language="text")
    
    # Add details
    with st.expander("View Infection Details", expanded=False):
        for event in events:
            if event.get("infected"):
                st.markdown(f"""
                **{event.get('agent')}**
                - Confidence: {event.get('confidence', 0):.1%}
                - Mutation: {'Yes' if event.get('mutation_detected') else 'No'}
                - Depth: {event.get('propagation_depth', 1)}
                """)
