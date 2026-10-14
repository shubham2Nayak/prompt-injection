"""
Propagation graph visualization using NetworkX and Plotly
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Any
import math


def create_propagation_graph(events: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive propagation graph.
    
    Args:
        events: List of propagation events
        
    Returns:
        Plotly figure
    """
    
    if not events:
        fig = go.Figure()
        fig.add_annotation(text="No propagation data available")
        return fig
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges based on propagation
    agents_seen = set()
    for i, event in enumerate(events):
        agent = event.get("agent", f"Agent-{i}")
        agents_seen.add(agent)
        
        if i > 0:
            prev_agent = events[i - 1].get("agent", f"Agent-{i-1}")
            G.add_edge(prev_agent, agent)
        
        G.add_node(agent, infected=event.get("infected", False), confidence=event.get("confidence", 0))
    
    # Add initial PDF node
    if agents_seen:
        G.add_node("PDF Document", infected=False, confidence=1.0)
        first_agent = events[0].get("agent", list(agents_seen)[0])
        G.add_edge("PDF Document", first_agent)
    
    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_colors = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
        # Check if edge represents infected propagation
        target_node = edge[1]
        target_infected = G.nodes[target_node].get("infected", False)
        edge_color = "#FF3333" if target_infected else "#0066FF"
        edge_colors.append(edge_color)
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        is_infected = G.nodes[node].get("infected", False)
        node_color = "#FF3333" if is_infected else "#00DD00" if node == "PDF Document" else "#0066FF"
        node_colors.append(node_color)
        
        confidence = G.nodes[node].get("confidence", 0)
        size = 30 + (confidence * 20)
        node_sizes.append(size)
        
        node_text.append(f"<b>{node}</b><br>{'Infected' if is_infected else 'Safe'}<br>Confidence: {confidence:.0%}")
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    for i in range(0, len(edge_x) - 2, 3):
        fig.add_trace(go.Scatter(
            x=[edge_x[i], edge_x[i + 1]],
            y=[edge_y[i], edge_y[i + 1]],
            mode='lines',
            line=dict(
                width=3,
                color=edge_colors[i // 3] if i // 3 < len(edge_colors) else "#0066FF",
            ),
            hoverinfo='none',
            showlegend=False,
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=11, color='white'),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False,
    ))
    
    fig.update_layout(
        title="Propagation Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    
    return fig


def render_propagation_graph(events: List[Dict[str, Any]]):
    """
    Render the propagation graph visualization.
    
    Args:
        events: List of propagation events
    """
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = create_propagation_graph(events)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Legend")
        st.markdown("""
        **Node Colors:**
        - 🟢 Green = Safe/Start
        - 🔴 Red = Infected
        - 🔵 Blue = Processing
        
        **Edge Colors:**
        - 🔵 Blue = Safe path
        - 🔴 Red = Infected path
        
        **Node Size:**
        - Larger = Higher confidence
        """)
        
        if events:
            infected = sum(1 for e in events if e.get("infected"))
            if infected > 0:
                st.error(f"⚠️ {infected} agent(s) infected")
            else:
                st.success("✅ Pipeline clean")
