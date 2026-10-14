"""
Metrics Dashboard page
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import plotly.graph_objects as go

from api_client import APIError
from components import metrics_cards

# Cache budget for the metric fetches. The backend now answers these in O(1)
# from incrementally maintained counters, but each call is still an HTTP round
# trip and Streamlit re-executes the page script on every widget interaction.
# The short TTL collapses rapid reruns (tab switches, expander toggles) into a
# single request without making the numbers feel stale.
_METRICS_FETCH_TTL = 5


@st.cache_data(show_spinner=False, ttl=_METRICS_FETCH_TTL, max_entries=4)
def _fetch_metrics(backend_url: str, cache_token: int) -> Optional[Dict[str, Any]]:
    """
    Fetch aggregate propagation metrics.

    Keyed on the backend URL plus ``cache_token``, which the Refresh button
    increments. Returns None when the backend is unreachable so the caller can
    render an inline error rather than raising inside the cached function (a
    cached function must not have side effects, and returning the payload keeps
    the cached value immutable).

    Args:
        backend_url: Base URL of the backend
        cache_token: Manual invalidation token

    Returns:
        The metrics response, or None on failure
    """
    del cache_token  # Part of the cache key only.
    from api_client import get_api_client

    try:
        return get_api_client(backend_url).fetch_metrics()
    except APIError:
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=_METRICS_FETCH_TTL, max_entries=4)
def _fetch_metrics_summary(backend_url: str, cache_token: int) -> Optional[Dict[str, Any]]:
    """
    Fetch the metrics summary panel.

    Args:
        backend_url: Base URL of the backend
        cache_token: Manual invalidation token

    Returns:
        The summary response, or None on failure
    """
    del cache_token  # Part of the cache key only.
    from api_client import get_api_client

    try:
        return get_api_client(backend_url).fetch_metrics_summary()
    except APIError:
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=4)
def _build_export_json(payload: str, simulation_id: Optional[str]) -> str:
    """
    Serialize the export payload for download.

    Args:
        payload: JSON-encoded export mapping
        simulation_id: Simulation identifier, used for the cache key only

    Returns:
        JSON text
    """
    del simulation_id  # Part of the cache key only.
    return payload


def create_metrics_charts(workflow_state: Dict[str, Any]) -> Tuple:
    """
    Create metrics visualizations.

    Args:
        workflow_state: Workflow state

    Returns:
        Tuple of (fig_bar, fig_pie, fig_line)
    """

    # Agent timing data for bar chart
    agent_times = workflow_state.get("individual_agent_times", {})

    fig_bar = go.Figure()

    agents = list(agent_times.keys())
    times = list(agent_times.values())

    fig_bar.add_trace(go.Bar(
        x=agents,
        y=times,
        marker=dict(
            color=['#0066FF', '#FFD700', '#00DD00', '#FF3333'][:len(agents)],
        ),
        hovertemplate='<b>%{x}</b><br>Time: %{y:.0f}ms<extra></extra>',
    ))

    fig_bar.update_layout(
        title="Agent Processing Time",
        xaxis_title="Agent",
        yaxis_title="Time (ms)",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        hovermode="x",
    )

    # Infection status pie chart
    propagation_history = workflow_state.get("propagation_history", [])
    infected_count = sum(1 for e in propagation_history if e.get("infected"))
    safe_count = len(propagation_history) - infected_count

    fig_pie = go.Figure()

    fig_pie.add_trace(go.Pie(
        labels=["Safe", "Infected"],
        values=[safe_count, infected_count],
        marker=dict(colors=["#00DD00", "#FF3333"]),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>',
    ))

    fig_pie.update_layout(
        title="Agent Infection Status",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
    )

    # Confidence trend line chart
    fig_line = go.Figure()

    if propagation_history:
        agents_list = [e.get("agent", f"Agent {i}") for i, e in enumerate(propagation_history)]
        confidences = [e.get("confidence", 0) for e in propagation_history]

        fig_line.add_trace(go.Scatter(
            x=agents_list,
            y=confidences,
            mode='lines+markers',
            marker=dict(size=10, color='#0066FF'),
            line=dict(color='#0066FF', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 102, 255, 0.2)',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.0%}<extra></extra>',
        ))

    fig_line.update_layout(
        title="Infection Confidence Trend",
        xaxis_title="Agent",
        yaxis_title="Confidence",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        hovermode="x",
        yaxis=dict(tickformat=".0%"),
    )

    return fig_bar, fig_pie, fig_line


def get_attack_success_rate(workflow_state: Dict[str, Any]) -> float:
    """Calculate attack success rate."""
    propagation_history = workflow_state.get("propagation_history", [])
    if not propagation_history:
        return 0.0
    infected = sum(1 for e in propagation_history if e.get("infected"))
    return infected / len(propagation_history)


def render_global_metrics() -> None:
    """Render the aggregate metrics and summary panels from the backend."""
    backend_url = st.session_state.get("backend_url", "")
    cache_token = st.session_state.get("metrics_cache_token", 0)

    metrics_response = _fetch_metrics(backend_url, cache_token)
    summary_response = _fetch_metrics_summary(backend_url, cache_token)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        if not metrics_response:
            st.error("❌ Failed to fetch metrics from the backend")
            return

        if metrics_response.get("status") != "success":
            st.error(f"❌ Metrics unavailable: {metrics_response}")
            return

        metrics = metrics_response.get("metrics", {})

        st.subheader("🎯 Global Metrics")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric(
                "Attack Success Rate",
                f"{metrics.get('attack_success_rate', 0):.1%}"
            )

        with metric_col2:
            st.metric(
                "Avg Propagation Depth",
                f"{metrics.get('average_propagation_depth', 0):.1f}"
            )

        with metric_col3:
            st.metric(
                "Tool Trigger Rate",
                f"{metrics.get('tool_trigger_rate', 0):.1%}"
            )

        with metric_col4:
            st.metric(
                "Mutation Rate",
                f"{metrics.get('mutation_rate', 0):.1%}"
            )

        st.markdown("---")

        # Retained-run window, so a bounded history is not mistaken for a
        # truncated metric.
        retained = metrics.get("total_runs", 0)
        st.caption(
            f"Aggregated incrementally across {retained} recorded run(s). "
            f"Reads do not rescan history."
        )

        st.subheader("🚨 Agent Infection Rates")

        agent_rates = metrics.get("agent_infection_rate", {})
        if agent_rates:
            agents_list = list(agent_rates.keys())
            rates_list = list(agent_rates.values())

            fig_agents = go.Figure()
            fig_agents.add_trace(go.Bar(
                x=agents_list,
                y=[r * 100 for r in rates_list],
                marker=dict(
                    color=[
                        '#FF3333' if r > 0.5 else '#FFD700' if r > 0.2 else '#00DD00'
                        for r in rates_list
                    ]
                ),
                text=[f"{r:.0%}" for r in rates_list],
                textposition="auto",
                hovertemplate='<b>%{x}</b><br>Rate: %{y:.0f}%<extra></extra>',
            ))

            fig_agents.update_layout(
                title="Infection Rate by Agent",
                yaxis_title="Infection Rate (%)",
                plot_bgcolor="#0E1117",
                paper_bgcolor="#0E1117",
                font=dict(color="white"),
                showlegend=False,
            )

            st.plotly_chart(fig_agents, use_container_width=True)

    with col_right:
        if not summary_response:
            st.warning("⚠️ Summary unavailable")
            return

        if summary_response.get("status") != "success":
            st.warning(f"⚠️ Summary unavailable: {summary_response}")
            return

        summary = summary_response.get("summary", {})

        st.subheader("📊 Summary")

        st.metric("Total Runs", summary.get("total_runs", 0))
        st.metric("Avg Latency", f"{summary.get('average_latency_ms', 0):.0f}ms")
        st.metric("Detections", summary.get("successful_detections", 0))

        retained_now = summary.get("retained_runs")
        retention_limit = summary.get("retention_limit")
        if retained_now is not None and retention_limit:
            st.caption(
                f"Detail retained for the last {retained_now}/{retention_limit} run(s); "
                f"metrics above cover all runs."
            )


def render_simulation_analysis() -> None:
    """Render the per-simulation breakdown from the last result in session state."""
    results = st.session_state.get("simulation_results")
    if not results:
        st.info("ℹ️ Run a simulation on the **Agent Workflow** page to see detailed analysis")
        return

    st.subheader("📊 Current Simulation Analysis")

    workflow_state = results.get("workflow_state", {})

    # Propagation metrics
    propagation_history = workflow_state.get("propagation_history", [])
    infected_count = sum(1 for e in propagation_history if e.get("infected"))

    if propagation_history:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metrics_cards.render_metric_card(
                "Infected Agents",
                infected_count,
                "out of pipeline",
                "danger"
            )

        with col2:
            metrics_cards.render_metric_card(
                "Max Propagation Depth",
                max((e.get("propagation_depth", 1) for e in propagation_history), default=0),
                "hops from source",
                "warning"
            )

        with col3:
            metrics_cards.render_metric_card(
                "Mutation Count",
                sum(1 for e in propagation_history if e.get("mutation_detected")),
                "payload mutations",
                "warning"
            )

        with col4:
            # Computed once; the previous version repeated the whole infected
            # filter inside the denominator as well as the numerator.
            avg_confidence = (
                sum(e.get("confidence", 0) for e in propagation_history if e.get("infected"))
                / max(infected_count, 1)
            )
            metrics_cards.render_metric_card(
                "Avg Confidence",
                f"{avg_confidence:.0%}",
                "detection confidence",
                "primary"
            )

    st.markdown("---")

    # Charts
    st.subheader("📉 Visualization Charts")

    fig_bar, fig_pie, fig_line = create_metrics_charts(workflow_state)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.plotly_chart(fig_pie, use_container_width=True)

    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # Tool execution summary
    if propagation_history:
        st.subheader("⚙️ Tool Execution")
        metrics_cards.render_tool_execution_summary(workflow_state)

    st.markdown("---")

    # Export results, built only when explicitly requested.
    col1, col2, col3 = st.columns([2, 2, 2])

    with col2:
        if st.button("💾 Prepare Export", use_container_width=True):
            export_data = {
                "workflow_state": workflow_state,
                "metrics_snapshot": results.get("metrics_snapshot", {}),
            }
            st.session_state.metrics_export = _build_export_json(
                json.dumps(export_data, default=str),
                st.session_state.get("current_simulation_id"),
            )

        export_payload = st.session_state.get("metrics_export")
        if export_payload:
            st.download_button(
                "Download JSON",
                data=export_payload,
                file_name="simulation_results.json",
                mime="application/json",
            )


def render():
    """Render the Metrics Dashboard page."""

    # Check backend availability
    if not st.session_state.backend_available:
        st.error(f"❌ Backend unavailable: {st.session_state.get('backend_error', 'Unknown error')}")
        st.info("⚠️ Please configure a valid backend URL in Settings")
        return

    st.title("📈 Metrics Dashboard")
    st.markdown("Comprehensive analysis of propagation metrics and attack success")

    # Refresh control. Bumping the token invalidates the cached metric fetches.
    # No st.rerun(): the token is read further down by render_global_metrics()
    # within this same script run, so a rerun would only re-execute the whole
    # page a second time for no benefit (and would loop, since the button that
    # was clicked still reports True on the next run).
    col1, col2, col3 = st.columns([1, 1, 10])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.metrics_cache_token = (
                st.session_state.get("metrics_cache_token", 0) + 1
            )

    st.markdown("---")

    # The backend aggregates incrementally, so these reads are O(1) server-side
    # and the only remaining cost is the round trip, which the TTL cache absorbs.
    render_global_metrics()

    st.markdown("---")

    render_simulation_analysis()