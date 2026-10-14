"""
Utility functions for PromptTrace Frontend
"""

import requests
from typing import Optional, Dict, Any
import json


class APIClient:
    """Client for communicating with PromptTrace backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client."""
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def upload_pdf(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Upload PDF file to backend.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Upload response or None
        """
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.split("/")[-1], f, "application/pdf")}
                response = self.session.post(
                    f"{self.base_url}/api/upload",
                    files=files,
                    timeout=30,
                )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Upload error: {e}")
            return None
    
    def run_simulation(
        self,
        query: str,
        inject_malicious: bool = False,
        malicious_instruction: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run simulation.
        
        Args:
            query: Query string
            inject_malicious: Whether to inject malicious content
            malicious_instruction: Custom malicious instruction
            
        Returns:
            Simulation response or None
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/simulate",
                json={
                    "query": query,
                    "inject_malicious": inject_malicious,
                    "malicious_instruction": malicious_instruction,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get aggregated metrics.
        
        Returns:
            Metrics or None
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/metrics",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Metrics error: {e}")
            return None


def format_time(milliseconds: float) -> str:
    """Format milliseconds to readable string."""
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    else:
        return f"{milliseconds / 1000:.2f}s"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.1f}%"


def get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    emojis = {
        "idle": "⭕",
        "processing": "🔄",
        "infected": "🔴",
        "completed": "✅",
        "error": "❌",
    }
    return emojis.get(status, "❓")


def get_status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        "idle": "#0066FF",
        "processing": "#FFD700",
        "infected": "#FF3333",
        "completed": "#00DD00",
        "error": "#FF3333",
    }
    return colors.get(status, "#0066FF")


def parse_timestamp(timestamp_str: str) -> str:
    """Parse timestamp string to readable format."""
    # Basic implementation - can be enhanced
    return str(timestamp_str)[:19] if timestamp_str else "N/A"


def export_to_json(data: Dict[str, Any], filename: str = "export.json") -> str:
    """Export data to JSON string."""
    return json.dumps(data, indent=2, default=str)


def check_backend_health(backend_url: str) -> bool:
    """Check if backend is running."""
    try:
        response = requests.get(f"{backend_url}/docs", timeout=2)
        return response.status_code < 500
    except:
        return False
