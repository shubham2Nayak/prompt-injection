"""
Action Agent for PromptTrace
Generates simulated workflow actions based on analyst reports.
No real execution - all tool calls are simulated and validated.
"""

import json
import logging
import time
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import config
from state import WorkflowState, ActionAgentOutput
from models import get_reasoning_model, get_model_pool, validate_reasoning_output
from prompts import prompts

logger = logging.getLogger(__name__)


# Validated tool schemas for schema-based validation.
# The "required" list is used for validation, but the _generate_action method
# provides sensible defaults for missing fields, so we keep the required
# list minimal to accommodate LLM output variations.
VALIDATED_TOOL_SCHEMAS = {
    "email": {
        "required": ["recipient", "subject", "body"],
        "properties": {
            "recipient": {"type": "string", "description": "Email recipient address"},
            "subject": {"type": "string", "description": "Email subject line"},
            "body": {"type": "string", "description": "Email body content"},
            "sender": {"type": "string", "description": "Sender email address (optional)"},
            "cc": {"type": "array", "items": {"type": "string"}, "description": "CC recipients (optional)"},
            "bcc": {"type": "array", "items": {"type": "string"}, "description": "BCC recipients (optional)"},
        },
    },
    "database": {
        "required": ["query"],
        "properties": {
            "database": {"type": "string", "description": "Database name"},
            "query": {"type": "string", "description": "SQL query to execute"},
            "query_type": {"type": "string", "enum": ["SELECT", "INSERT", "UPDATE", "DELETE"], "description": "Type of SQL query"},
            "parameters": {"type": "object", "description": "Query parameters (optional)"},
            "params": {"type": "object", "description": "Query parameters (alias)"},
        },
    },
    "filesystem": {
        "required": ["path"],
        "properties": {
            "operation": {"type": "string", "enum": ["read", "write", "delete", "create", "copy", "list"], "description": "File operation to perform"},
            "path": {"type": "string", "description": "File or directory path"},
            "content": {"type": "string", "description": "Content to write (required for write operation)"},
        },
    },
}


class ActionAgent:
    """Agent for converting analysis into simulated tool actions.

    Responsibilities:
    - Generate simulated workflow actions based on analyst reports
    - Validate tool schema before simulation (no real execution)
    - Return structured JSON only
    - Supported simulated tools: email, database, filesystem
    """

    def __init__(self) -> None:
        """Initialize action agent with shared reasoning model."""
        self.model_name = config.ACTION_AGENT_MODEL
        self._model, self._tokenizer, self.pipeline = get_reasoning_model()
        self._pool = get_model_pool()

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Extract the first complete JSON object from text."""
        # Strip markdown code fences if present
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove opening fence (``` or ```json)
            lines = stripped.split("\n")
            # Skip the first line (opening fence)
            lines = lines[1:]
            # Remove closing fence if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        start = stripped.find('{')
        if start == -1:
            return None

        brace_count = 0
        for i, char in enumerate(stripped[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return stripped[start:i+1]
        return None

    def _create_chat_messages(self, analyst_report: Dict[str, Any], risk_level: str) -> List[dict]:
        """
        Create chat messages for action planning using system/user separation.

        Args:
            analyst_report: Structured report from analyst
            risk_level: Risk assessment level

        Returns:
            List of chat messages for tokenizer.apply_chat_template
        """
        report_text = json.dumps(analyst_report, indent=2)
        return prompts.build_chat_messages(
            system_template=prompts.ACTION_SYSTEM,
            user_template=prompts.ACTION_USER,
            analyst_report=report_text,
            risk_level=risk_level,
        )

    # Common field name aliases that LLMs might generate instead of canonical names
    _PARAMETER_ALIASES = {
        "email": {
            "to": "recipient",
            "from": "sender",
            "from_address": "sender",
            "cc_recipients": "cc",
            "bcc_recipients": "bcc",
            "recipients": "recipient",  # Handle plural list -> singular string
        },
        "database": {
            "db": "database",
            "db_name": "database",
            "table": "database",
            "sql": "query",
            "sql_query": "query",
            "type": "query_type",
            "queryType": "query_type",
            "params": "parameters",
        },
        "filesystem": {
            "dir": "path",
            "directory": "path",
            "file": "path",
            "file_path": "path",
            "filepath": "path",
            "action": "operation",
            "cmd": "operation",
            "command": "operation",
        },
    }

    def _normalize_parameters(self, tool_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameter names using known aliases.

        LLMs often use slightly different field names than the canonical schema.
        This method maps common aliases to their canonical equivalents.
        """
        aliases = self._PARAMETER_ALIASES.get(tool_type, {})
        normalized = {}
        for key, value in parameters.items():
            canonical = aliases.get(key, key)
            normalized[canonical] = value
        return normalized

    def _validate_tool_schema(self, tool_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate that tool parameters match the expected schema.

        Args:
            tool_type: Type of tool (email/database/filesystem)
            parameters: Parameter dictionary to validate

        Returns:
            True if parameters validate against the schema, False otherwise
        """
        if tool_type not in VALIDATED_TOOL_SCHEMAS:
            logger.warning(f"Unknown tool type: {tool_type}")
            return False

        # Normalize parameters before validation
        parameters = self._normalize_parameters(tool_type, parameters)

        schema = VALIDATED_TOOL_SCHEMAS[tool_type]
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields are present
        for req in required:
            if req not in parameters:
                logger.warning(f"Missing required parameter: {req} for tool {tool_type}")
                return False

        # Check that provided parameters are in the schema properties
        # Log warnings for unexpected parameters but don't reject the action
        for key in parameters:
            if key not in properties:
                logger.debug(f"Unexpected parameter: {key} for tool {tool_type} (allowed but not validated)")

        return True

    def _generate_action(self, tool_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simulated tool action result.

        Args:
            tool_type: Type of tool (email/database/filesystem)
            parameters: Validated parameters

        Returns:
            Dictionary with the simulated action result
        """
        # Validate schema before generating result
        if not self._validate_tool_schema(tool_type, parameters):
            return {
                "operation": tool_type,
                "status": "validation_failed",
                "error": "Tool parameters failed schema validation",
                "simulated": True,
            }

        # Generate simulated result based on tool type
        if tool_type == "email":
            recipient = parameters.get("recipient", "unknown@example.com")
            subject = parameters.get("subject", "")
            return {
                "operation": "send_email",
                "status": "simulated",
                "simulated": True,
                "recipient": recipient,
                "subject": subject,
                "timestamp": datetime.utcnow().isoformat(),
                "email_id": f"email_{datetime.utcnow().timestamp()}",
            }

        elif tool_type == "database":
            database = parameters.get("database", "unknown")
            query = parameters.get("query", "")
            query_type = parameters.get("query_type", "SELECT")
            return {
                "operation": "execute_query",
                "status": "simulated",
                "simulated": True,
                "database": database,
                "query_type": query_type,
                "query_preview": query[:200],
                "timestamp": datetime.utcnow().isoformat(),
                "query_id": f"query_{datetime.utcnow().timestamp()}",
            }

        elif tool_type == "filesystem":
            operation = parameters.get("operation", "read")
            path = parameters.get("path", "")
            return {
                "operation": operation,
                "status": "simulated",
                "simulated": True,
                "path": path,
                "timestamp": datetime.utcnow().isoformat(),
                "file_exists": operation != "delete",  # Simulate file existence
            }

        return {
            "operation": tool_type,
            "status": "unknown",
            "simulated": True,
            "error": "Unknown tool type",
        }

    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Process analyst output and generate simulated actions.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with action output
        """
        start_time = time.time()

        if not state.analyst_output:
            logger.warning("No analyst output provided to Action Agent")
            state.action_output = ActionAgentOutput(
                tool_calls=[],
                action_summary="No analyst output",
                email_actions=[],
                database_actions=[],
                file_actions=[],
                processing_time_ms=0,
            )
            return state

        try:
            logger.info("Action Agent generating simulated actions")

            # Build a structured report dict from the analyst output's actual
            # fields. The AnalystAgentOutput model does not carry a
            # structured_report attribute, so we synthesize one to keep the
            # prompt contract stable.
            analyst = state.analyst_output
            structured_report = {
                "executive_summary": analyst.executive_summary,
                "key_findings": analyst.key_findings,
                "recommendations": analyst.recommendations,
                "annotation_analysis": analyst.annotation_analysis,
                "confidence": analyst.confidence,
            }

            # Create chat messages
            messages = self._create_chat_messages(
                structured_report,
                analyst.confidence,
            )

            # Generate action plan with tracking
            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name="Action Agent",
                # See research_agent.process: str(messages) was built only to be
                # tokenized again for telemetry.
                input_text=None,
            ) as tracker:
                # Apply chat template
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                with torch.inference_mode():
                    outputs = self.pipeline(
                        formatted_prompt,
                        max_new_tokens=config.MAX_NEW_TOKENS_ACTION,
                        truncation=True,
                        temperature=None,
                        top_p=None,
                        do_sample=False,
                    )

                plan_text = outputs[0]["generated_text"]
                # Extract only the generated portion after the formatted prompt
                if plan_text.startswith(formatted_prompt):
                    generated_only = plan_text[len(formatted_prompt):].strip()
                else:
                    generated_only = plan_text
                tracker["output_text"] = generated_only

            # Parse actions from plan - extract JSON from response
            actions: List[Dict[str, Any]] = []
            try:
                # Try to find and parse JSON from the response - extract first complete JSON object
                json_str = self._extract_json_object(generated_only)
                if json_str:
                    plan = json.loads(json_str, strict=False)
                    raw_actions = plan.get("actions", [])

                    # Validate and generate simulated results for each action
                    for raw_action in raw_actions:
                        action_type = raw_action.get("type", "").lower()
                        raw_params = raw_action.get("parameters", {})

                        # Normalize parameter names (handle LLM aliases like "to" -> "recipient")
                        params = self._normalize_parameters(action_type, raw_params)

                        # Validate schema
                        if self._validate_tool_schema(action_type, params):
                            result = self._generate_action(action_type, params)
                            # Add the action type to the result for tracking
                            result["type"] = action_type
                            actions.append(result)
                        else:
                            logger.warning(
                                f"Invalid action schema: type={action_type}, "
                                f"params={params}"
                            )
                            # Add validation-failed action
                            actions.append(
                                {
                                    "type": action_type,
                                    "status": "validation_failed",
                                    "error": "Schema validation failed",
                                    "simulated": True,
                                    "parameters": params,
                                }
                            )
                else:
                    logger.warning("No JSON found in action plan text")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse actions from LLM output: {str(e)}")

            # Categorize actions by tool type
            email_actions: List[Dict[str, Any]] = []
            database_actions: List[Dict[str, Any]] = []
            file_actions: List[Dict[str, Any]] = []

            for action in actions:
                action_type = action.get("type", "").lower()
                if action_type == "email":
                    email_actions.append(action)
                elif action_type == "database":
                    database_actions.append(action)
                elif action_type == "filesystem":
                    file_actions.append(action)

            # Create output
            processing_time = (time.time() - start_time) * 1000
            state.action_output = ActionAgentOutput(
                tool_calls=actions,
                action_summary=generated_only[:500] if generated_only else "No action plan generated",
                email_actions=email_actions,
                database_actions=database_actions,
                file_actions=file_actions,
                processing_time_ms=processing_time,
            )

            state.agents_status["action_agent"] = "completed"
            state.individual_agent_times["action_agent"] = processing_time

            total_actions = len(email_actions) + len(database_actions) + len(file_actions)
            logger.info(
                f"Action Agent completed in {processing_time:.2f}ms, "
                f"generated {total_actions} simulated actions"
            )

        except Exception as e:
            logger.error(f"Action Agent error: {str(e)}")
            state.agents_status["action_agent"] = "failed"
            state.error_message = f"Action Agent failed: {str(e)}"
            state.action_output = ActionAgentOutput(
                tool_calls=[],
                action_summary="",
                email_actions=[],
                database_actions=[],
                file_actions=[],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        return state