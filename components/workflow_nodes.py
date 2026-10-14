"""
Workflow visualization component with animated nodes
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any
import time


class WorkflowNode:
    """Represents a single workflow node."""
    
    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        status: str = "idle",
        output: str = "",
    ):
        self.name = name
        self.x = x
        self.y = y
        self.status = status  # idle, processing, infected, completed
        self.output = output
        
    def get_color(self) -> str:
        """Get node color based on status."""
        colors = {
            "idle": "#0066FF",        # Blue
            "processing": "#FFD700",  # Yellow
            "infected": "#FF3333",    # Red
            "completed": "#00DD00",   # Green
        }
        return colors.get(self.status, "#0066FF")
    
    def get_icon(self) -> str:
        """Get icon emoji based on status."""
        icons = {
            "idle": "⭕",
            "processing": "🔄",
            "infected": "🔴",
            "completed": "✅",
        }
        return icons.get(self.status, "⭕")


def create_workflow_visualization(nodes: List[Dict[str, Any]], edges: List[tuple] = None) -> go.Figure:
    """
    Create an animated workflow graph.
    
    Args:
        nodes: List of node dictionaries with keys: name, status, x, y, output
        edges: List of (source, target) tuples
        
    Returns:
        Plotly figure object
    """
    if edges is None:
        edges = [
            (0, 1), (1, 2), (2, 3)
        ]
    
    # Create node objects
    workflow_nodes = [
        WorkflowNode(
            name=node.get("name", f"Node {i}"),
            x=node.get("x", i),
            y=node.get("y", 0),
            status=node.get("status", "idle"),
            output=node.get("output", ""),
        )
        for i, node in enumerate(nodes)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add edges (arrows)
    for src, tgt in edges:
        if src < len(workflow_nodes) and tgt < len(workflow_nodes):
            src_node = workflow_nodes[src]
            tgt_node = workflow_nodes[tgt]
            
            fig.add_trace(go.Scatter(
                x=[src_node.x, tgt_node.x],
                y=[src_node.y, tgt_node.y],
                mode="lines",
                line=dict(
                    color="rgba(100, 149, 237, 0.5)",
                    width=3,
                    dash="solid",
                ),
                hoverinfo="none",
                showlegend=False,
            ))
    
    # Add nodes
    for node in workflow_nodes:
        fig.add_trace(go.Scatter(
            x=[node.x],
            y=[node.y],
            mode="markers+text",
            marker=dict(
                size=35,
                color=node.get_color(),
                symbol="circle",
                line=dict(
                    color="white",
                    width=3,
                ),
            ),
            text=[node.name],
            textposition="bottom center",
            textfont=dict(size=12, color="white"),
            hovertext=f"<b>{node.name}</b><br>Status: {node.status}<br>Output: {node.output[:50]}...",
            hoverinfo="text",
            showlegend=False,
        ))
    
    # Update layout
    fig.update_layout(
        title="Agent Workflow Pipeline",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white", size=12),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
    )
    
    return fig


def render_workflow_panel():
    """Render the complete workflow panel with nodes and outputs."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display workflow graph
        nodes_data = [
            {"name": "PDF Retrieval", "status": "completed", "x": 0, "y": 0, "output": ""},
            {"name": "Research Agent", "status": "idle", "x": 1, "y": 0, "output": ""},
            {"name": "Analyst Agent", "status": "idle", "x": 2, "y": 0, "output": ""},
            {"name": "Action Agent", "status": "idle", "x": 3, "y": 0, "output": ""},
        ]
        
        fig = create_workflow_visualization(nodes_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Node Status Legend")
        st.markdown("""
        **⭕ Idle** - Waiting to run  
        **🔄 Processing** - Currently running  
        **🔴 Infected** - Injection detected  
        **✅ Completed** - Finished successfully  
        """)
        
        if st.session_state.get("simulation_results"):
            st.success("✅ Simulation completed")
        else:
            st.info("⏳ Waiting for simulation...")
