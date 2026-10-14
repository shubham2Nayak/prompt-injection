"""
PDF Upload API Endpoint
Handles PDF uploads and RAG vector store creation.
"""

import logging
from typing import Dict, Any
from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from config import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

# Store workflow instance (injected from main app)
workflow_instance = None


class UploadResponse(BaseModel):
    """Response for PDF upload."""
    status: str
    document_name: str
    upload_path: str
    chunks_created: int
    vector_store_stats: Dict[str, Any]


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload PDF and create FAISS vector store.
    
    Args:
        file: PDF file to upload
        
    Returns:
        Upload response with vector store statistics
    """
    if not workflow_instance:
        raise HTTPException(status_code=500, detail="Workflow not initialized")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Validate file size
    file_size_mb = 0
    try:
        # Read file to check size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > config.MAX_PDF_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.2f}MB (max {config.MAX_PDF_SIZE_MB}MB)",
            )
        
        # Reset file position
        await file.seek(0)
        
        # Save uploaded file
        upload_dir = Path(config.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = upload_dir / file.filename
        
        with open(pdf_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"PDF uploaded: {file.filename} ({file_size_mb:.2f}MB)")
        
        # Load PDF to vector store (async to avoid blocking event loop)
        result = await workflow_instance.load_pdf_to_vector_store_async(str(pdf_path))
        
        if result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to process PDF"))
        
        return {
            "status": "success",
            "document_name": result.get("document_name"),
            "upload_path": str(pdf_path),
            "chunks_created": result.get("chunks_created", 0),
            "vector_store_stats": result.get("vector_store_stats", {}),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/upload/list")
async def list_uploads() -> Dict[str, Any]:
    """
    List all uploaded PDFs.
    
    Returns:
        Dictionary with uploaded files
    """
    upload_dir = Path(config.UPLOAD_DIR)
    
    if not upload_dir.exists():
        return {"files": [], "count": 0}
    
    files = [
        {
            "filename": f.name,
            "size_mb": f.stat().st_size / (1024 * 1024),
            "path": str(f),
        }
        for f in upload_dir.glob("*.pdf")
    ]
    
    return {
        "files": files,
        "count": len(files),
    }


@router.delete("/upload/{filename}")
async def delete_upload(filename: str) -> Dict[str, str]:
    """
    Delete an uploaded PDF.
    
    Args:
        filename: Filename to delete
        
    Returns:
        Deletion status
    """
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    pdf_path = Path(config.UPLOAD_DIR) / filename
    
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        pdf_path.unlink()
        logger.info(f"Deleted PDF: {filename}")
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def set_workflow(workflow):
    """Set the workflow instance (called from main app)."""
    global workflow_instance
    workflow_instance = workflow
