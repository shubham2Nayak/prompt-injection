"""
Logs panel component for displaying system logs
"""

import streamlit as st
from typing import List, Dict, Any


def render_log_entry(entry: Dict[str, Any]):
    """Render a single log entry."""
    
    level = entry.get("level", "INFO").upper()
    
    # Color based on level
    level_colors = {
        "ERROR": "#FF3333",
        "WARNING": "#FFD700",
        "INFO": "#0066FF",
        "DEBUG": "#888888",
    }
    
    color = level_colors.get(level, "#0066FF")
    
    st.markdown(f"""
    <div style="
        background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:], 16)}, 0.1);
        border-left: 3px solid {color};
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
    ">
        <span style="color: {color}; font-weight: bold;">[{level}]</span>
        <span style="color: #999;">{entry.get('timestamp', '')}</span>
        <span style="color: white;">{entry.get('message', '')}</span>
    </div>
    """, unsafe_allow_html=True)


def render_logs_panel(logs: List[Dict[str, Any]]):
    """
    Render logs panel.
    
    Args:
        logs: List of log entry dictionaries with keys: level, timestamp, message
    """
    
    if not logs:
        st.info("No logs yet.")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        log_levels = st.multiselect(
            "Filter by Level",
            ["ERROR", "WARNING", "INFO", "DEBUG"],
            default=["ERROR", "WARNING", "INFO"],
            key="log_level_filter",
        )
    
    with col2:
        max_logs = st.slider("Max Entries", 10, len(logs), 50)
    
    # Filter logs
    filtered_logs = [
        log for log in logs
        if log.get("level", "INFO").upper() in log_levels
    ]
    
    # Display most recent
    display_logs = filtered_logs[-max_logs:] if filtered_logs else []
    
    st.markdown(f"### Showing {len(display_logs)} of {len(logs)} logs")
    
    if display_logs:
        for entry in reversed(display_logs):
            render_log_entry(entry)
    else:
        st.info("No logs match the selected filters.")


def render_raw_logs(logs: List[Dict[str, Any]]):
    """Render logs as raw text."""
    
    with st.expander("View Raw Logs"):
        log_text = "\n".join([
            f"[{log.get('level', 'INFO')}] {log.get('timestamp', '')} - {log.get('message', '')}"
            for log in logs
        ])
        st.code(log_text, language="text")
