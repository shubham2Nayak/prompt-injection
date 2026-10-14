"""
Retrieved chunks display component
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any


def render_chunk_card(chunk: Dict[str, Any], index: int):
    """Render a single chunk card."""
    
    # Determine trust level color
    trust_level = chunk.get("trust_level", "untrusted")
    trust_color = "🟢" if trust_level == "trusted" else "🔴"
    
    with st.expander(
        f"{trust_color} Chunk {chunk.get('chunk_id', index)} - Page {chunk.get('page_number', 'N/A')}",
        expanded=(index == 0)
    ):
        cols = st.columns([2, 1, 1])
        
        with cols[0]:
            st.caption(f"**Source:** {chunk.get('document_name', 'Unknown')}")
        
        with cols[1]:
            st.caption(f"**ID:** {chunk.get('chunk_id', 'N/A')}")
        
        with cols[2]:
            st.caption(f"**Trust:** {trust_level.capitalize()}")
        
        st.markdown("---")
        
        # Chunk text
        st.markdown("**Content:**")
        st.text(chunk.get("retrieved_text", "No content"))


def render_retrieved_chunks(chunks: List[Dict[str, Any]]):
    """
    Render retrieved chunks from RAG retrieval.
    
    Args:
        chunks: List of chunk dictionaries
    """
    
    if not chunks:
        st.info("No chunks retrieved yet. Run a simulation to see retrieved content.")
        return
    
    st.subheader("📄 Retrieved Document Chunks")
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Chunks", len(chunks))
    
    with col2:
        trusted_count = sum(1 for c in chunks if c.get("trust_level") == "trusted")
        st.metric("Trusted Chunks", trusted_count)
    
    with col3:
        untrusted_count = len(chunks) - (sum(1 for c in chunks if c.get("trust_level") == "trusted"))
        st.metric("Untrusted Chunks", untrusted_count)
    
    st.markdown("---")
    
    # Display chunks
    for i, chunk in enumerate(chunks):
        render_chunk_card(chunk, i)


def render_chunks_table(chunks: List[Dict[str, Any]]):
    """
    Render chunks as a searchable table.
    
    Args:
        chunks: List of chunk dictionaries
    """
    
    if not chunks:
        return
    
    with st.expander("View as Table", expanded=False):
        df = pd.DataFrame(chunks)
        
        # Select display columns
        display_cols = [
            "chunk_id",
            "document_name",
            "page_number",
            "trust_level",
            "chunk_index",
        ]
        
        available_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[available_cols],
            use_container_width=True,
            hide_index=True,
        )


def render_document_preview(document_metadata: Dict[str, Any]):
    """
    Render document metadata preview.
    
    Args:
        document_metadata: Document metadata dictionary
    """
    
    with st.expander("📋 Document Metadata", expanded=True):
        cols = st.columns(2)
        
        with cols[0]:
            st.metric("Document Name", document_metadata.get("document_name", "Unknown"))
            st.metric("File Size", f"{document_metadata.get('file_size_mb', 0):.2f} MB")
        
        with cols[1]:
            st.metric("Total Pages", document_metadata.get("total_pages", 0))
            st.metric("Chunks Created", document_metadata.get("chunks_created", 0))
        
        if document_metadata.get("upload_time"):
            st.caption(f"Uploaded: {document_metadata.get('upload_time')}")
