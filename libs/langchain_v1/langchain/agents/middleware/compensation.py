"""Compensation middleware for agents with automatic LIFO rollback."""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict

from langchain_core.messages import AIMessage, ToolMessage

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest


class CompensationRecord(dict):
    """Record of a compensatable action stored in the compensation stack."""

    def __init__(
        self,
        *,
        id: str,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        timestamp: float,
        compensated: bool = False,
        compensation_tool: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        super().__init__(
            id=id,
            tool_name=tool_name,
            params=params,
            result=result,
            timestamp=timestamp,
            compensated=compensated,
            compensation_tool=compensation_tool,
            metadata=metadata or {},
        )


class CompensationStack:
    """Stack to track compensatable actions in LIFO order."""

    def __init__(self) -> None:
        self._stack: list[CompensationRecord] = []
        self._compensated_ids: set[str] = set()

    def push(self, record: CompensationRecord) -> None:
        """Push a new compensatable action onto the stack."""
        self._stack.append(record)

    def pop(self) -> CompensationRecord | None:
        """Pop and return the most recent uncompensated action."""
        while self._stack:
            record = self._stack.pop()
            if not record.get("compensated", False):
                return record
        return None

    def mark_compensated(self, record_id: str) -> None:
        """Mark a specific action as compensated."""
        self._compensated_ids.add(record_id)
        for record in self._stack:
            if record.get("id") == record_id:
                record["compensated"] = True

    def get_uncompensated(self) -> list[CompensationRecord]:
        """Get all uncompensated actions in LIFO order (most recent first)."""
        return [r for r in reversed(self._stack) if not r.get("compensated", False)]

    def clear(self) -> None:
        """Clear the entire compensation stack."""
        self._stack.clear()
        self._compensated_ids.clear()

    def __len__(self) -> int:
        """Return the number of uncompensated actions."""
        return len([r for r in self._stack if not r.get("compensated", False)])


class CompensationState(dict):
    """State extension for compensation tracking."""

    def __init__(self):
        super().__init__(
            compensation_stack=CompensationStack(),
            compensated_actions=[],
        )


class CompensationMiddleware(AgentMiddleware[CompensationState, Any]):
    """Middleware that provides automatic compensation for tools with side effects.

    This middleware transparently handles compensation without adding nodes to the graph.
    When a tool execution fails, all previously successful compensatable actions are
    automatically rolled back in LIFO (Last-In-First-Out) order.

    The middleware integrates seamlessly into the agent's existing tool execution flow,
    making compensation invisible to end users while ensuring data consistency.

    Examples:
        !!! example "Basic compensation setup"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import CompensationMiddleware

            # Define compensation mapping
            compensation_mapping = {
                "book_flight": "cancel_flight",
                "book_hotel": "cancel_hotel",
                "reserve_car": "cancel_car_reservation",
            }

            middleware = CompensationMiddleware(
                compensation_mapping=compensation_mapping
            )

            agent = create_agent(
                model="openai:gpt-4",
                tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
                middleware=[middleware]
            )
            ```

        !!! example "With custom parameter mapping"

            ```python
            def map_booking_to_cancellation(result: dict, params: dict) -> dict:
                '''Extract booking ID from result for cancellation.'''
                return {"booking_id": result.get("id")}

            middleware = CompensationMiddleware(
                compensation_mapping={
                    "book_flight": "cancel_flight",
                    "book_hotel": "cancel_hotel",
                },
                state_mappers={
                    "book_flight": map_booking_to_cancellation,
                    "book_hotel": map_booking_to_cancellation,
                }
            )
            ```

        !!! example "Automatic parameter detection"

            ```python
            # If no state_mapper provided, middleware automatically extracts
            # common parameter names from the result
            middleware = CompensationMiddleware(
                compensation_mapping={
                    "create_resource": "delete_resource",  # Auto-detects 'id'
                    "allocate_budget": "deallocate_budget",  # Auto-detects 'transaction_id'
                }
            )
            ```

    Args:
        compensation_mapping: Dictionary mapping tool names to their compensation tools.
            For example: `{"book_flight": "cancel_flight"}`.
        state_mappers: Optional dictionary of custom parameter mapping functions.
            Each function takes `(result, params)` and returns the parameters for
            the compensation tool. If not provided, automatic parameter detection is used.
        name: Optional name for the middleware instance. Defaults to `"compensation"`.
    """

    state_schema = CompensationState

    def __init__(
        self,
        *,
        compensation_mapping: Dict[str, str],
        state_mappers: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]]
        | None = None,
        name: str = "compensation",
    ):
        """Initialize the compensation middleware.

        Args:
            compensation_mapping: Maps tool names to their compensation tools.
            state_mappers: Optional parameter mapping functions per tool.
            name: Middleware instance name.
        """
        self.compensation_mapping = compensation_mapping
        self.state_mappers = state_mappers or {}
        self._name = name
        self._tools_by_name: Dict[str, Any] = {}  # Cache for tool instances

    @property
    def name(self) -> str:
        """Return the middleware name."""
        return self._name

    def _get_tool(self, tool_name: str, request: ToolCallRequest) -> Any | None:
        """Get a tool instance by name from the runtime.

        Args:
            tool_name: Name of the tool to retrieve.
            request: The tool call request containing runtime context.

        Returns:
            The tool instance if found, None otherwise.
        """
        # Check cache first
        if tool_name in self._tools_by_name:
            return self._tools_by_name[tool_name]

        # Try to get tools from the runtime's graph
        runtime = request.runtime
        if hasattr(runtime, "graph"):
            graph = runtime.graph
            # Look for the tools node in the graph
            if hasattr(graph, "nodes") and "tools" in graph.nodes:
                tools_node = graph.nodes["tools"]
                if hasattr(tools_node, "tools_by_name"):
                    tools_by_name = tools_node.tools_by_name
                    # Cache all tools for future use
                    self._tools_by_name.update(tools_by_name)
                    return tools_by_name.get(tool_name)

        return None

    def _extract_tool_result(self, msg: ToolMessage) -> Any:
        """Extract the actual result from a ToolMessage content."""
        content = msg.content
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            # Try to parse JSON
            if content.startswith("{") and content.endswith("}"):
                try:
                    return json.loads(content)
                except Exception:  # noqa: BLE001
                    return content
        return content

    def _is_error_result(self, result: ToolMessage) -> bool:
        """Check if a tool result indicates an error.

        Args:
            result: The ToolMessage to check.

        Returns:
            True if the result indicates an error, False otherwise.
        """
        # Check status attribute
        if hasattr(result, "status") and result.status == "error":
            return True

        # Check content for error indicators (case-insensitive)
        # Handle both string content and dict content with 'message' field
        content = result.content
        if isinstance(content, dict):
            # Check if dict has 'status' field indicating error
            if content.get("status") == "error":
                return True
            # Check if dict has 'message' field with error text
            if "message" in content:
                content = content["message"]
            # Check for error in dict keys or values
            elif any("error" in str(k).lower() or "error" in str(v).lower()
                     for k, v in content.items()):
                return True

        content_str = str(content).lower()
        error_indicators = [
            "error:",
            "error",
            "failed",
            "exception",
            "traceback",
            "not available",
            "unavailable",
            "cannot",
            "unable to",
            "does not exist",
        ]
        return any(indicator in content_str for indicator in error_indicators)

    def _map_compensation_params(
        self,
        record: CompensationRecord,
        comp_tool_name: str,
        request: "ToolCallRequest | None" = None,
    ) -> Dict[str, Any]:
        """Map parameters from the original action to the compensation tool.

        Args:
            record: The compensation record containing original params and result.
            comp_tool_name: Name of the compensation tool.
            request: Optional tool call request to access runtime tools.

        Returns:
            Dictionary of parameters for the compensation tool.
        """
        tool_name = record["tool_name"]

        # Use custom mapper if provided
        if tool_name in self.state_mappers:
            mapper = self.state_mappers[tool_name]
            try:
                return mapper(record["result"], record["params"])
            except Exception:  # noqa: BLE001
                pass

        # Automatic parameter detection
        result = record["result"]

        # Try to match parameters based on compensation tool schema
        if request and isinstance(result, dict):
            comp_tool = self._get_tool(comp_tool_name, request)
            if comp_tool:
                arg_names = set()

                # Try to get the Pydantic model first
                schema = getattr(comp_tool, "args_schema", None)
                if not schema and hasattr(comp_tool, "get_input_schema"):
                     try:
                        schema = comp_tool.get_input_schema()
                     except Exception:  # noqa: BLE001
                        pass

                if schema and not isinstance(schema, dict):
                     if hasattr(schema, "model_fields"):  # Pydantic v2
                        arg_names = set(schema.model_fields.keys())
                     elif hasattr(schema, "__fields__"):  # Pydantic v1
                        arg_names = set(schema.__fields__.keys())

                if not arg_names:
                    # Fallback to args property
                    args_dict = getattr(comp_tool, "args", None)
                    if isinstance(args_dict, dict):
                         arg_names = set(args_dict.keys())

                # If we found argument names, look for them in the result or original params
                if arg_names:
                    mapped_params = {}
                    for arg in arg_names:
                        if arg in result:
                            mapped_params[arg] = result[arg]
                        elif arg in record["params"]:
                            mapped_params[arg] = record["params"][arg]

                    if mapped_params:
                        return mapped_params

        if isinstance(result, dict):
            # Try common ID field names
            for key in [
                "id",
                "booking_id",
                "resource_id",
                "transaction_id",
                "reservation_id",
                "order_id",
            ]:
                if key in result:
                    return {key: result[key]}

            # Try to match any key from result that might be an identifier
            for key, value in result.items():
                if "_id" in key.lower() or key.lower() == "id":
                    return {key: value}

        return {}

    def _execute_compensation_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        request: ToolCallRequest,
    ) -> ToolMessage:
        """Execute a compensation tool.

        Args:
            tool_name: Name of the compensation tool to execute.
            params: Parameters for the compensation tool.
            request: The original tool call request (for context).

        Returns:
            ToolMessage with the compensation result.
        """
        tool_call_id = str(uuid.uuid4())

        try:
            # Get the actual tool instance
            tool = self._get_tool(tool_name, request)

            if tool is None:
                msg = f"Compensation tool '{tool_name}' not found in available tools"
                return ToolMessage(
                    content=f"Error: {msg}",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )

            # Invoke the tool with the mapped parameters
            result = tool.invoke(params)

            # Return successful ToolMessage
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        except Exception as e:  # noqa: BLE001
            return ToolMessage(
                content=f"Compensation failed: {e}",
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
            )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Wrap tool execution with compensation tracking.

        This is the core sync method that handles:
        1. Tracking successful compensatable actions
        2. Detecting failures
        3. Executing compensation in LIFO order when failures occur

        Args:
            request: The tool call request to execute.
            handler: The base tool execution handler.

        Returns:
            The tool execution result, potentially with compensation applied.
        """
        tool_name = request.tool_call["name"]
        state = request.state

        # Get or initialize compensation stack
        if "compensation_stack" not in state or state["compensation_stack"] is None:
            state["compensation_stack"] = CompensationStack()
        comp_stack: CompensationStack = state["compensation_stack"]  # type: ignore[assignment]

        # Execute the tool
        result = handler(request)

        # Only process ToolMessage results (not Command)
        if not isinstance(result, ToolMessage):
            return result

        # Check if this tool is compensatable
        is_compensatable = tool_name in self.compensation_mapping

        # Check if execution failed
        is_error = self._is_error_result(result)

        if is_error:
            # Failure detected - trigger compensation in LIFO order
            compensated_actions = []
            compensation_messages = []

            for record in comp_stack.get_uncompensated():
                comp_tool_name = record.get("compensation_tool")
                if not comp_tool_name:
                    continue

                # Map parameters for compensation
                comp_params = self._map_compensation_params(record, comp_tool_name, request)

                if comp_params:
                    # Execute compensation tool
                    comp_result = self._execute_compensation_tool(
                        comp_tool_name, comp_params, request
                    )

                    # Mark as compensated
                    comp_stack.mark_compensated(record["id"])

                    compensated_actions.append({
                        "original_tool": record["tool_name"],
                        "compensation_tool": comp_tool_name,
                        "result": comp_result.content,
                    })

                    # Add compensation result to messages
                    compensation_messages.append(comp_result)

            # Update state with compensation info
            state["compensation_stack"] = comp_stack
            state["compensated_actions"] = state.get("compensated_actions", []) + compensated_actions

            # Inject compensation messages into the conversation history
            if compensation_messages:
                # Add the compensation ToolMessages to the state's message list
                if "messages" in state:
                    state["messages"] = list(state["messages"]) + compensation_messages

            return result

        elif is_compensatable:
            # Successful execution of a compensatable action - track it
            record = CompensationRecord(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                params=request.tool_call.get("args", {}),
                result=self._extract_tool_result(result),
                timestamp=time.time(),
                compensated=False,
                compensation_tool=self.compensation_mapping[tool_name],
            )
            comp_stack.push(record)
            state["compensation_stack"] = comp_stack

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call.

        Provides the same compensation logic as the sync version but for async execution.

        Args:
            request: The tool call request to execute.
            handler: The async base tool execution handler.

        Returns:
            The tool execution result, potentially with compensation applied.
        """
        tool_name = request.tool_call["name"]
        state = request.state

        # Get or initialize compensation stack
        if "compensation_stack" not in state or state["compensation_stack"] is None:
            state["compensation_stack"] = CompensationStack()
        comp_stack: CompensationStack = state["compensation_stack"]  # type: ignore[assignment]

        # Execute the tool
        result = await handler(request)

        # Only process ToolMessage results (not Command)
        if not isinstance(result, ToolMessage):
            return result

        # Check if this tool is compensatable
        is_compensatable = tool_name in self.compensation_mapping

        # Check if execution failed
        is_error = self._is_error_result(result)

        if is_error:
            # Failure detected - trigger compensation in LIFO order
            compensated_actions = []
            compensation_messages = []

            for record in comp_stack.get_uncompensated():
                comp_tool_name = record.get("compensation_tool")
                if not comp_tool_name:
                    continue

                # Map parameters for compensation
                comp_params = self._map_compensation_params(record, comp_tool_name, request)

                if comp_params:
                    # Execute compensation tool
                    comp_result = self._execute_compensation_tool(
                        comp_tool_name, comp_params, request
                    )

                    # Mark as compensated
                    comp_stack.mark_compensated(record["id"])

                    compensated_actions.append({
                        "original_tool": record["tool_name"],
                        "compensation_tool": comp_tool_name,
                        "result": comp_result.content,
                    })

                    # Add compensation result to messages
                    compensation_messages.append(comp_result)

            # Update state with compensation info
            state["compensation_stack"] = comp_stack
            state["compensated_actions"] = state.get("compensated_actions", []) + compensated_actions

            # Inject compensation messages into the conversation history
            if compensation_messages:
                # Add the compensation ToolMessages to the state's message list
                if "messages" in state:
                    state["messages"] = list(state["messages"]) + compensation_messages

            return result

        elif is_compensatable:
            # Successful execution of a compensatable action - track it
            record = CompensationRecord(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                params=request.tool_call.get("args", {}),
                result=self._extract_tool_result(result),
                timestamp=time.time(),
                compensated=False,
                compensation_tool=self.compensation_mapping[tool_name],
            )
            comp_stack.push(record)
            state["compensation_stack"] = comp_stack

        return result


__all__ = [
    "CompensationMiddleware",
    "CompensationRecord",
    "CompensationStack",
    "CompensationState",
]
