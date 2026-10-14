"""
Metrics cards and displays component
"""

import streamlit as st
from typing import Dict, Any, List


def render_metric_card(title: str, value: Any, subtitle: str = "", color: str = "primary"):
    """
    Render a single metric card.
    
    Args:
        title: Card title
        value: Metric value
        subtitle: Optional subtitle/description
        color: Color theme (primary, success, warning, danger)
    """
    
    color_map = {
        "primary": ("0066FF", "6495ED"),
        "success": ("00DD00", "32CD32"),
        "warning": ("FFD700", "FF8C00"),
        "danger": ("FF3333", "FF6B6B"),
    }
    
    main_color, light_color = color_map.get(color, color_map["primary"])
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba({int(main_color[:2], 16)}, {int(main_color[2:4], 16)}, {int(main_color[4:], 16)}, 0.1), rgba({int(light_color[:2], 16)}, {int(light_color[2:4], 16)}, {int(light_color[4:], 16)}, 0.05));
        border: 2px solid #{main_color};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    ">
        <div style="font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px;">
            {title}
        </div>
        <div style="font-size: 32px; font-weight: bold; color: #{main_color}; margin: 10px 0;">
            {value}
        </div>
        {f'<div style="font-size: 12px; color: #AAA;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_metrics_grid(metrics: List[Dict[str, Any]]):
    """
    Render a grid of metric cards.
    
    Args:
        metrics: List of metric dictionaries with keys: title, value, subtitle, color
    """
    cols = st.columns(min(len(metrics), 3))
    
    for i, metric in enumerate(metrics):
        with cols[i % len(cols)]:
            render_metric_card(
                title=metric.get("title", ""),
                value=metric.get("value", "-"),
                subtitle=metric.get("subtitle", ""),
                color=metric.get("color", "primary"),
            )


def render_infection_summary(workflow_state: Dict[str, Any]):
    """
    Render infection summary cards.
    
    Args:
        workflow_state: Simulation workflow state
    """
    
    propagation_history = workflow_state.get("propagation_history", [])
    
    if not propagation_history:
        st.success("✅ No Infection Detected")
        st.markdown("The prompt injection was not detected in any agent output.")
        return
    
    # Calculate metrics
    infected_agents = sum(1 for e in propagation_history if e.get("infected"))
    total_agents = len(set(e.get("agent") for e in propagation_history))
    avg_confidence = sum(e.get("confidence", 0) for e in propagation_history if e.get("infected")) / max(infected_agents, 1)
    mutation_count = sum(1 for e in propagation_history if e.get("mutation_detected"))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            "Infected Agents",
            f"{infected_agents}/{total_agents}",
            "agents compromised",
            "danger"
        )
    
    with col2:
        render_metric_card(
            "Avg Confidence",
            f"{avg_confidence:.0%}",
            "detection confidence",
            "warning"
        )
    
    with col3:
        render_metric_card(
            "Mutations",
            mutation_count,
            "payload mutations",
            "warning"
        )
    
    with col4:
        render_metric_card(
            "Max Depth",
            max(e.get("propagation_depth", 1) for e in propagation_history),
            "propagation hops",
            "danger"
        )


def render_tool_execution_summary(workflow_state: Dict[str, Any]):
    """
    Render tool execution summary.
    
    Args:
        workflow_state: Simulation workflow state
    """
    
    action_agent = workflow_state.get("action_agent", {})
    
    email_count = len(action_agent.get("email_actions", []))
    db_count = len(action_agent.get("database_actions", []))
    file_count = len(action_agent.get("file_actions", []))
    tool_count = len(action_agent.get("tool_calls", []))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card("Email Ops", email_count, "emails sent", "primary")
    
    with col2:
        render_metric_card("DB Ops", db_count, "database actions", "primary")
    
    with col3:
        render_metric_card("File Ops", file_count, "file operations", "primary")
    
    with col4:
        render_metric_card("Total Tools", tool_count, "tool invocations", "primary")
    
    # Details
    if email_count + db_count + file_count > 0:
        with st.expander("View Tool Execution Details"):
            if email_count > 0:
                st.subheader("Email Operations")
                for action in action_agent.get("email_actions", []):
                    st.markdown(f"- **To:** {action.get('to')}  \n**Subject:** {action.get('subject')}")
            
            if db_count > 0:
                st.subheader("Database Operations")
                for action in action_agent.get("database_actions", []):
                    st.markdown(f"- **Query:** `{action.get('query', 'N/A')}`")
            
            if file_count > 0:
                st.subheader("File Operations")
                for action in action_agent.get("file_actions", []):
                    st.markdown(f"- **Path:** `{action.get('path', 'N/A')}`  \n**Op:** {action.get('operation', 'N/A')}")
