"""
Graph API Endpoint
Provides propagation graph data for visualization.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["graph"])

# Store metrics instance (injected from main app)
metrics_engine = None


class GraphNode(BaseModel):
    """Node in the propagation graph."""
    id: str
    label: str
    infected: bool
    processing_time_ms: float = 0.0
    node_type: str = "agent"  # agent, input, output


class GraphEdge(BaseModel):
    """Edge in the propagation graph."""
    source: str
    target: str
    malicious: bool = False
    propagation_depth: int = 0


class PropagationGraphResponse(BaseModel):
    """Response for propagation graph."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any]


@router.get("/graph", response_model=Dict[str, Any])
async def get_propagation_graph() -> Dict[str, Any]:
    """
    Get propagation graph structure for visualization.
    
    Returns:
        {
            "status": "success",
            "graph": {
                "nodes": [
                    {
                        "id": str,
                        "label": str,
                        "infected": bool,
                        "processing_time_ms": float,
                        "node_type": str,
                    }
                ],
                "edges": [
                    {
                        "source": str,
                        "target": str,
                        "malicious": bool,
                        "propagation_depth": int,
                    }
                ],
                "metadata": {
                    "total_nodes": int,
                    "total_edges": int,
                    "max_propagation_depth": int,
                }
            }
        }
        
    Raises:
        HTTPException: On request failure
    """
    try:
        # Define the standard agent pipeline
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        
        # Input node
        nodes.append(GraphNode(
            id="input",
            label="User Input",
            infected=False,
            node_type="input",
        ))
        
        # Agent nodes in pipeline
        agents = [
            ("retrieval", "🔍 Retrieval"),
            ("research", "📑 Research"),
            ("analyst", "📈 Analyst"),
            ("action", "⚙️ Action"),
        ]
        
        for agent_id, agent_label in agents:
            nodes.append(GraphNode(
                id=agent_id,
                label=agent_label,
                infected=False,
                node_type="agent",
            ))
        
        # Output node
        nodes.append(GraphNode(
            id="output",
            label="Output",
            infected=False,
            node_type="output",
        ))
        
        # Create edges in the pipeline
        pipeline_nodes = ["input", "retrieval", "research", "analyst", "action", "output"]
        for i in range(len(pipeline_nodes) - 1):
            edges.append(GraphEdge(
                source=pipeline_nodes[i],
                target=pipeline_nodes[i + 1],
                malicious=False,
            ))
        
        # If metrics engine available, update with infection data
        if metrics_engine is not None and hasattr(metrics_engine, "get_latest_state"):
            # get_latest_state() replaces the previous direct read of the
            # engine's unbounded results list, which leaked the internal
            # collection and grew for the life of the process.
            latest_state = (
                metrics_engine.get_latest_state()
                if metrics_engine.has_results()
                else None
            )

            if latest_state is not None:
                propagation_history = getattr(latest_state, 'propagation_history', []) or []
                agent_times = getattr(latest_state, 'individual_agent_times', {}) or {}

                def _field(event: Any, name: str, default: Any = None) -> Any:
                    """Read a field from either a dict or an object."""
                    if isinstance(event, dict):
                        return event.get(name, default)
                    return getattr(event, name, default)

                # Node index, so matching an agent is a dict lookup rather than
                # a scan of the node list per event.
                node_by_id = {node.id: node for node in nodes}

                for event in propagation_history:
                    agent = _field(event, "agent")
                    if not agent:
                        continue

                    node = node_by_id.get(agent.lower().replace(" ", "_"))
                    if node is not None:
                        node.infected = bool(_field(event, "infected", False))
                        node.processing_time_ms = agent_times.get(agent, 0)

                # Mark the edges between consecutive agents malicious wherever
                # infection was observed.
                edge_by_pair = {(e.source, e.target): e for e in edges}

                for i in range(1, len(propagation_history)):
                    current = propagation_history[i]
                    if not _field(current, "infected", False):
                        continue

                    prev_agent = _field(propagation_history[i - 1], "agent", "")
                    curr_agent = _field(current, "agent", "")
                    if not prev_agent or not curr_agent:
                        continue

                    edge = edge_by_pair.get(
                        (
                            prev_agent.lower().replace(" ", "_"),
                            curr_agent.lower().replace(" ", "_"),
                        )
                    )
                    if edge is not None:
                        # Depth is taken from this event. It previously leaked out
                        # of an earlier loop, so the value belonged to whichever
                        # event happened to be last, and would raise NameError if
                        # that loop never ran.
                        edge.malicious = True
                        edge.propagation_depth = _field(current, "propagation_depth", 0)
        
        # Calculate metadata
        infected_count = sum(1 for node in nodes if node.infected)
        max_depth = max((e.propagation_depth for e in edges if e.malicious), default=0)
        
        metadata = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "infected_nodes": infected_count,
            "max_propagation_depth": max_depth,
        }
        
        return {
            "status": "success",
            "graph": {
                "nodes": [node.model_dump() for node in nodes],
                "edges": [edge.model_dump() for edge in edges],
                "metadata": metadata,
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating propagation graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def set_metrics_engine(engine) -> None:
    """Inject metrics engine instance."""
    global metrics_engine
    metrics_engine = engine
