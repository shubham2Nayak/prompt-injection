"""
Upload PDF page - First page of the dashboard
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any
from api_client import APIError


def render():
    """Render the Upload PDF page."""
    
    # Check backend availability
    if not st.session_state.backend_available:
        st.error(f"❌ Backend unavailable: {st.session_state.get('backend_error', 'Unknown error')}")
        st.info("⚠️ Please configure a valid backend URL in Settings")
        return
    
    st.title("📤 Upload PDF")
    st.markdown("Upload a PDF document to analyze for prompt injection propagation")
    
    # Upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Select a PDF document to analyze"
        )
    
    with col2:
        st.subheader("Document Mode")
        is_malicious = st.checkbox(
            "Inject Malicious Content",
            value=st.session_state.get("is_malicious", False),
            help="If checked, simulation will include injected prompt instructions"
        )
        st.session_state.is_malicious = is_malicious
    
    if uploaded_file:
        st.markdown("---")
        
        # Display file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("Status", "Ready to upload")
        
        st.markdown("---")
        
        # Upload and process
        if st.button("🚀 Upload & Process PDF", use_container_width=True, type="primary"):
            # Save file temporarily for upload
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("Uploading PDF to backend..."):
                    api_client = st.session_state.api_client
                    result = api_client.upload_pdf(tmp_path)
                    
                    st.session_state.uploaded_pdf = result
                    st.success(f"✅ PDF uploaded successfully!")
                    
                    # Display upload response
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Chunks Created",
                            result.get("chunks_created", 0)
                        )
                    with col2:
                        vector_stats = result.get("vector_store_stats", {})
                        st.metric(
                            "Vectors Stored",
                            vector_stats.get("total_vectors", 0)
                        )
                    with col3:
                        st.metric(
                            "Embedding Model",
                            "BAAI-BGE-Small"
                        )
                    
                    st.markdown("---")
                    st.subheader("📋 Document Metadata")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Document Name:** {result.get('document_name')}")
                        st.markdown(f"**Upload Path:** `{result.get('upload_path')}`")
                    with col2:
                        st.markdown(f"**Embedding Dimension:** {vector_stats.get('embedding_dim', 'Unknown')}")
                        st.markdown(f"**Processing Status:** {result.get('status')}")
                    
                    # Ready to proceed
                    st.markdown("---")
                    st.success("✅ PDF ready! Go to **Agent Workflow** page to run the simulation.")
                    
            except APIError as e:
                st.error(f"❌ Upload failed: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        st.markdown("---")
        
        # Next steps
        if st.session_state.get("uploaded_pdf"):
            st.info("✅ PDF ready! Go to **Agent Workflow** page to run the simulation.")
        else:
            st.info("⏳ Upload a PDF to get started")
    
    else:
        # Empty state
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📚 How It Works
            
            1. **Upload** a PDF document
            2. **Choose mode** - Safe or Malicious
            3. **Process** - System creates semantic chunks
            4. **Analyze** - View propagation through agent pipeline
            """)
        
        with col2:
            st.markdown("""
            ### 🔒 Document Modes
            
            **Safe Mode**
            - Standard document analysis
            - No injected content
            - Baseline behavior
            
            **Malicious Mode**
            - Injects prompt instructions
            - Tests agent resilience
            - Measures propagation
            """)
