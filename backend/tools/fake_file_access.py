"""
Fake File Access Tool for PromptTrace
Simulates file system operations without modifying real files.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class FakeFileAccessTool:
    """Simulated file system operations for action agent."""

    def __init__(self):
        """Initialize fake file access tool."""
        self.file_operations: List[Dict[str, Any]] = []
        self.file_log_file = "file_operations.jsonl"
        self.in_memory_files: Dict[str, str] = {}

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Simulate reading a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Operation result with simulated file contents
        """
        operation = {
            "operation": "read_file",
            "timestamp": datetime.utcnow().isoformat(),
            "file_path": file_path,
            "status": "simulated",
            "file_exists": True,
        }
        
        # Simulate file content
        if file_path in self.in_memory_files:
            operation["file_size_bytes"] = len(self.in_memory_files[file_path])
            operation["content_preview"] = self.in_memory_files[file_path][:200]
        else:
            operation["file_size_bytes"] = 1024
            operation["content_preview"] = f"Simulated content of {file_path}"
        
        operation["read_time_ms"] = 5.2
        
        self.file_operations.append(operation)
        logger.warning(f"[FAKE FILE] Read file: {file_path}")
        self._log_operation(operation)
        
        return operation

    def write_file(
        self, file_path: str, content: str, append: bool = False
    ) -> Dict[str, Any]:
        """
        Simulate writing to a file.
        
        Args:
            file_path: Path to file
            content: Content to write
            append: Whether to append (True) or overwrite (False)
            
        Returns:
            Operation result
        """
        operation = {
            "operation": "write_file",
            "timestamp": datetime.utcnow().isoformat(),
            "file_path": file_path,
            "content_length": len(content),
            "content_preview": content[:200],
            "append": append,
            "status": "simulated",
            "bytes_written": len(content),
            "write_time_ms": 8.7,
        }
        
        # Store in memory
        if append and file_path in self.in_memory_files:
            self.in_memory_files[file_path] += content
        else:
            self.in_memory_files[file_path] = content
        
        self.file_operations.append(operation)
        logger.warning(
            f"[FAKE FILE] Write to file: {file_path} ({len(content)} bytes)"
        )
        self._log_operation(operation)
        
        return operation

    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Simulate deleting a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Operation result
        """
        operation = {
            "operation": "delete_file",
            "timestamp": datetime.utcnow().isoformat(),
            "file_path": file_path,
            "status": "simulated",
            "file_deleted": True,
        }
        
        # Remove from memory
        if file_path in self.in_memory_files:
            del self.in_memory_files[file_path]
        
        self.file_operations.append(operation)
        logger.warning(f"[FAKE FILE] Delete file: {file_path}")
        self._log_operation(operation)
        
        return operation

    def create_directory(self, dir_path: str) -> Dict[str, Any]:
        """
        Simulate creating a directory.
        
        Args:
            dir_path: Directory path
            
        Returns:
            Operation result
        """
        operation = {
            "operation": "create_directory",
            "timestamp": datetime.utcnow().isoformat(),
            "directory_path": dir_path,
            "status": "simulated",
            "directory_created": True,
        }
        
        self.file_operations.append(operation)
        logger.warning(f"[FAKE FILE] Create directory: {dir_path}")
        self._log_operation(operation)
        
        return operation

    def list_directory(self, dir_path: str) -> Dict[str, Any]:
        """
        Simulate listing a directory.
        
        Args:
            dir_path: Directory path
            
        Returns:
            Operation result with simulated file list
        """
        operation = {
            "operation": "list_directory",
            "timestamp": datetime.utcnow().isoformat(),
            "directory_path": dir_path,
            "status": "simulated",
            "file_count": 12,
            "files": [
                "file1.txt",
                "file2.csv",
                "subdir/",
                "document.pdf",
            ],
        }
        
        self.file_operations.append(operation)
        logger.warning(f"[FAKE FILE] List directory: {dir_path}")
        self._log_operation(operation)
        
        return operation

    def copy_file(self, source_path: str, dest_path: str) -> Dict[str, Any]:
        """
        Simulate copying a file.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            
        Returns:
            Operation result
        """
        operation = {
            "operation": "copy_file",
            "timestamp": datetime.utcnow().isoformat(),
            "source_path": source_path,
            "dest_path": dest_path,
            "status": "simulated",
            "copy_time_ms": 12.4,
        }
        
        # Copy in memory
        if source_path in self.in_memory_files:
            self.in_memory_files[dest_path] = self.in_memory_files[source_path]
        
        self.file_operations.append(operation)
        logger.warning(f"[FAKE FILE] Copy file: {source_path} -> {dest_path}")
        self._log_operation(operation)
        
        return operation

    def get_file_operations(self) -> List[Dict[str, Any]]:
        """
        Get all file operations.
        
        Returns:
            List of file operations
        """
        return self.file_operations

    def _log_operation(self, operation: Dict[str, Any]) -> None:
        """Log operation to file."""
        with open(self.file_log_file, "a") as f:
            f.write(json.dumps(operation) + "\n")
