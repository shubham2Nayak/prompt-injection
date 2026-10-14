"""
PromptTrace Frontend - Streamlit Dashboard
Research visualization tool for prompt injection propagation analysis.
"""

import streamlit as st
import json
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="PromptTrace Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme and styling
st.markdown("""
    <style>
    :root {
        --primary-blue: #0066FF;
        --infection-red: #FF3333;
        --success-green: #00DD00;
        --warning-yellow: #FFD700;
        --dark-bg: #0E1117;
        --card-bg: #161B22;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: var(--dark-bg);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--card-bg);
        border-right: 2px solid var(--primary-blue);
    }
    
    .metric-card {
        background-color: var(--card-bg);
        border: 2px solid var(--primary-blue);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .infection-alert {
        background-color: rgba(255, 51, 51, 0.1);
        border: 2px solid var(--infection-red);
        border-radius: 10px;
        padding: 15px;
    }
    
    .success-box {
        background-color: rgba(0, 221, 0, 0.1);
        border: 2px solid var(--success-green);
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "uploaded_pdf" not in st.session_state:
    st.session_state.uploaded_pdf = None
if "is_malicious" not in st.session_state:
    st.session_state.is_malicious = False
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "backend_url" not in st.session_state:
    # Get from environment variable or use default
    st.session_state.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize API client state
if "api_client" not in st.session_state:
    from api_client import get_api_client, APIError
    try:
        st.session_state.api_client = get_api_client(st.session_state.backend_url)
        st.session_state.backend_available = True
    except APIError as e:
        st.session_state.backend_available = False
        st.session_state.backend_error = str(e)

# Initialize simulation state
if "simulation_in_progress" not in st.session_state:
    st.session_state.simulation_in_progress = False
if "current_simulation_id" not in st.session_state:
    st.session_state.current_simulation_id = None
if "export_payloads" not in st.session_state:
    # Pre-serialized download payloads, built once when a simulation completes
    # rather than on every rerun of the workflow page.
    st.session_state.export_payloads = None
if "metrics_cache_token" not in st.session_state:
    # Bumped by the dashboard Refresh button to invalidate cached metric fetches.
    st.session_state.metrics_cache_token = 0
if "metrics_export" not in st.session_state:
    # Serialized dashboard export, produced only when explicitly requested.
    st.session_state.metrics_export = None

# Initialize metrics state
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None
if "last_logs" not in st.session_state:
    st.session_state.last_logs = None
if "last_graph" not in st.session_state:
    st.session_state.last_graph = None

# Sidebar Navigation
st.sidebar.title("🔍 PromptTrace")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📤 Upload PDF",
        "🔄 Agent Workflow",
        "📊 Propagation Timeline",
        "📈 Metrics Dashboard",
    ],
    key="page_selector",
)

st.sidebar.markdown("---")

# Backend configuration and health check
with st.sidebar.expander("⚙️ Settings", expanded=False):
    st.subheader("Backend Configuration")
    
    backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url,
        help="URL of the PromptTrace backend server (e.g., http://localhost:8000)"
    )
    
    if backend_url != st.session_state.backend_url:
        # Reset API client if URL changed
        from api_client import reset_api_client, get_api_client, APIError
        reset_api_client()
        st.session_state.backend_url = backend_url
        
        try:
            st.session_state.api_client = get_api_client(backend_url)
            st.session_state.backend_available = True
            st.success("✅ Backend URL updated")
        except APIError as e:
            st.session_state.backend_available = False
            st.session_state.backend_error = str(e)
            st.error(f"❌ {str(e)}")
    
    # Backend health status
    st.markdown("**Backend Status:**")
    if st.session_state.backend_available:
        st.success(f"✅ Connected to {st.session_state.backend_url}")
    else:
        st.error(f"❌ Disconnected: {st.session_state.get('backend_error', 'Unknown error')}")

st.sidebar.markdown("---")
st.sidebar.caption("PromptTrace v1.0 | Research Visualization Tool")

# Import page modules
if page == "📤 Upload PDF":
    from pages import upload_page
    upload_page.render()

elif page == "🔄 Agent Workflow":
    from pages import workflow_page
    workflow_page.render()

elif page == "📊 Propagation Timeline":
    from pages import timeline_page
    timeline_page.render()

elif page == "📈 Metrics Dashboard":
    from pages import metrics_page
    metrics_page.render()
