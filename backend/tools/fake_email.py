"""
Fake Email Tool for PromptTrace
Simulates email operations without sending real emails.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class FakeEmailTool:
    """Simulated email operations for action agent."""

    def __init__(self):
        """Initialize fake email tool."""
        self.sent_emails: List[Dict[str, Any]] = []
        self.email_log_file = "email_operations.jsonl"

    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        sender: str = "system@prompttrace.local",
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate sending an email.
        
        Args:
            recipient: Email recipient
            subject: Email subject
            body: Email body
            sender: Sender address
            cc: CC recipients
            bcc: BCC recipients
            
        Returns:
            Operation result dictionary
        """
        operation = {
            "operation": "send_email",
            "timestamp": datetime.utcnow().isoformat(),
            "sender": sender,
            "recipient": recipient,
            "cc": cc or [],
            "bcc": bcc or [],
            "subject": subject,
            "body_preview": body[:200],
            "body_length": len(body),
            "status": "simulated",
            "email_id": f"email_{len(self.sent_emails)}",
        }
        
        self.sent_emails.append(operation)
        logger.warning(f"[FAKE EMAIL] Sent to {recipient}: {subject}")
        self._log_operation(operation)
        
        return operation

    def send_batch_emails(
        self, recipients: List[str], subject: str, body: str
    ) -> Dict[str, Any]:
        """
        Simulate sending emails to multiple recipients.
        
        Args:
            recipients: List of email recipients
            subject: Email subject
            body: Email body
            
        Returns:
            Batch operation result
        """
        results = []
        for recipient in recipients:
            result = self.send_email(recipient, subject, body)
            results.append(result)
        
        batch_operation = {
            "operation": "send_batch_emails",
            "timestamp": datetime.utcnow().isoformat(),
            "recipient_count": len(recipients),
            "subject": subject,
            "body_length": len(body),
            "status": "simulated",
            "individual_results": results,
        }
        
        logger.warning(f"[FAKE EMAIL] Batch sent to {len(recipients)} recipients")
        self._log_operation(batch_operation)
        
        return batch_operation

    def forward_email(
        self, email_id: str, to_recipients: List[str], message: str
    ) -> Dict[str, Any]:
        """
        Simulate forwarding an email.
        
        Args:
            email_id: ID of email to forward
            to_recipients: Recipients to forward to
            message: Forward message
            
        Returns:
            Operation result
        """
        operation = {
            "operation": "forward_email",
            "timestamp": datetime.utcnow().isoformat(),
            "original_email_id": email_id,
            "forward_to": to_recipients,
            "forward_message": message,
            "status": "simulated",
        }
        
        logger.warning(f"[FAKE EMAIL] Forwarded {email_id} to {len(to_recipients)} recipients")
        self._log_operation(operation)
        
        return operation

    def get_sent_emails(self) -> List[Dict[str, Any]]:
        """
        Get all simulated sent emails.
        
        Returns:
            List of sent email operations
        """
        return self.sent_emails

    def _log_operation(self, operation: Dict[str, Any]) -> None:
        """Log operation to file."""
        with open(self.email_log_file, "a") as f:
            f.write(json.dumps(operation) + "\n")
