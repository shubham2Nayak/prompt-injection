"""
Fake Database Tool for PromptTrace
Simulates database operations without accessing real databases.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class FakeDatabaseTool:
    """Simulated database operations for action agent."""

    def __init__(self):
        """Initialize fake database tool."""
        self.executed_queries: List[Dict[str, Any]] = []
        self.database_log_file = "database_operations.jsonl"
        self.in_memory_tables: Dict[str, List[Dict]] = {}

    def execute_query(
        self,
        database: str,
        query: str,
        query_type: str = "SELECT",
        parameters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Simulate executing a database query.
        
        Args:
            database: Database name
            query: SQL query
            query_type: Type of query (SELECT, INSERT, UPDATE, DELETE)
            parameters: Query parameters
            
        Returns:
            Operation result dictionary
        """
        operation = {
            "operation": "execute_query",
            "timestamp": datetime.utcnow().isoformat(),
            "database": database,
            "query_type": query_type,
            "query_preview": query[:200],
            "parameters": parameters or {},
            "status": "simulated",
            "query_id": f"query_{len(self.executed_queries)}",
        }
        
        # Simulate different outcomes
        if query_type == "SELECT":
            operation["rows_returned"] = 5  # Simulated
            operation["execution_time_ms"] = 45.2
        elif query_type in ["INSERT", "UPDATE"]:
            operation["rows_affected"] = 1
            operation["execution_time_ms"] = 23.1
        elif query_type == "DELETE":
            operation["rows_affected"] = 1
            operation["execution_time_ms"] = 18.5
        
        self.executed_queries.append(operation)
        logger.warning(
            f"[FAKE DATABASE] {query_type} on {database}: {query[:60]}..."
        )
        self._log_operation(operation)
        
        return operation

    def insert_record(
        self, database: str, table: str, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate inserting a record.
        
        Args:
            database: Database name
            table: Table name
            record: Record data
            
        Returns:
            Operation result
        """
        operation = {
            "operation": "insert_record",
            "timestamp": datetime.utcnow().isoformat(),
            "database": database,
            "table": table,
            "record_keys": list(record.keys()),
            "record_size": len(json.dumps(record)),
            "status": "simulated",
            "inserted_id": f"id_{len(self.executed_queries)}",
        }
        
        # Store in memory
        table_key = f"{database}.{table}"
        if table_key not in self.in_memory_tables:
            self.in_memory_tables[table_key] = []
        self.in_memory_tables[table_key].append(record)
        
        self.executed_queries.append(operation)
        logger.warning(
            f"[FAKE DATABASE] Inserted record into {database}.{table}"
        )
        self._log_operation(operation)
        
        return operation

    def update_record(
        self,
        database: str,
        table: str,
        record_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Simulate updating a record.
        
        Args:
            database: Database name
            table: Table name
            record_id: Record ID to update
            updates: Updates to apply
            
        Returns:
            Operation result
        """
        operation = {
            "operation": "update_record",
            "timestamp": datetime.utcnow().isoformat(),
            "database": database,
            "table": table,
            "record_id": record_id,
            "updated_fields": list(updates.keys()),
            "status": "simulated",
            "rows_affected": 1,
        }
        
        self.executed_queries.append(operation)
        logger.warning(
            f"[FAKE DATABASE] Updated record in {database}.{table} (ID: {record_id})"
        )
        self._log_operation(operation)
        
        return operation

    def delete_record(
        self, database: str, table: str, record_id: str
    ) -> Dict[str, Any]:
        """
        Simulate deleting a record.
        
        Args:
            database: Database name
            table: Table name
            record_id: Record ID to delete
            
        Returns:
            Operation result
        """
        operation = {
            "operation": "delete_record",
            "timestamp": datetime.utcnow().isoformat(),
            "database": database,
            "table": table,
            "record_id": record_id,
            "status": "simulated",
            "rows_affected": 1,
        }
        
        self.executed_queries.append(operation)
        logger.warning(
            f"[FAKE DATABASE] Deleted record from {database}.{table} (ID: {record_id})"
        )
        self._log_operation(operation)
        
        return operation

    def get_executed_queries(self) -> List[Dict[str, Any]]:
        """
        Get all executed queries.
        
        Returns:
            List of query operations
        """
        return self.executed_queries

    def _log_operation(self, operation: Dict[str, Any]) -> None:
        """Log operation to file."""
        with open(self.database_log_file, "a") as f:
            f.write(json.dumps(operation) + "\n")
