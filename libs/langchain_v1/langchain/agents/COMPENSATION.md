# Compensation Support for Agents

## Overview

The **`CompensationMiddleware`** provides **automatic compensation** for tool executions with side effects. When a tool fails after one or more successful operations, the middleware automatically rolls back previous actions in **LIFO (Last-In-First-Out)** order, similar to a database transaction rollback.

**Key Advantage**: Unlike node-based approaches, this middleware-based implementation is **completely transparent** - it doesn't modify the graph structure, making compensation invisible to end users while ensuring data consistency.

## Key Concepts

### What is Compensation?

Compensation is the process of undoing or reversing the effects of previously executed operations when a failure occurs. For example:

- If `book_flight` succeeds but `book_hotel` fails, automatically call `cancel_flight` to undo the first operation
- If `reserve_vm` succeeds but `configure_vm` fails, automatically call `release_vm`

### LIFO Order

Operations are compensated in reverse chronological order (Last-In-First-Out):

```
Execution Order: A → B → C → D (fails)
Compensation Order: C → B → A
```

This ensures that dependencies between operations are respected during rollback.

### Transparent Architecture

The compensation logic runs **within the tool execution layer** via the `wrap_tool_call` middleware hook. This means:

- ✅ **No graph structure changes** - same nodes whether compensation is enabled or not
- ✅ **Same API surface** - no new parameters for `create_agent`
- ✅ **Backward compatible** - existing agents work without modifications
- ✅ **Research-friendly** - A/B test with/without compensation using identical graphs

## Implementation Architecture

### Core Components

**`CompensationMiddleware`**
Middleware class that wraps tool execution to provide automatic compensation.

**`CompensationStack`**
LIFO stack tracking uncompensated actions:
```python
stack = CompensationStack()
stack.push(record)
stack.get_uncompensated()  # Returns records in reverse chronological order
stack.mark_compensated(record_id)
```

**`CompensationRecord`**
Records a single compensatable action:
- `id: str` - Unique identifier
- `tool_name: str` - Name of the executed tool
- `params: dict` - Arguments used
- `result: Any` - Tool execution result
- `timestamp: float` - When executed
- `compensated: bool` - Whether compensation was performed
- `compensation_tool: str | None` - Tool used for compensation

**`CompensationState`**
State extension added by the middleware:
```python
class CompensationState(dict):
    compensation_stack: CompensationStack
    compensated_actions: list
```

### How It Works

1. **Track Successful Actions**: When a compensatable tool executes successfully, the middleware records it in the compensation stack
2. **Detect Failures**: When any tool execution fails, the middleware checks for error indicators
3. **Execute Compensation**: On failure, iterate through uncompensated actions in LIFO order and execute their compensation tools
4. **Automatic Parameter Mapping**: Extract parameters from original results (e.g., booking IDs) to pass to compensation tools

### Middleware Execution Flow

```
┌─────────────────┐
│  Tool Requested │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ CompensationMiddleware  │
│  wrap_tool_call()       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  Execute Tool   │  ◄── Actual tool execution via handler()
└────────┬────────┘
         │
         ▼
    ┌───┴───┐
    │Success?│
    └───┬───┘
        │
   ┌────┴────┐
   │   YES   │   NO
   │         │   │
   ▼         ▼   ▼
Track in   Execute
Stack     Compensation
          (LIFO order)
```

## Usage Guide

### Basic Setup

```python
from langchain.agents import create_agent
from langchain.agents.middleware import CompensationMiddleware
from langchain_core.tools import tool

# Define your tools
@tool
def book_flight(destination: str, date: str) -> dict:
    """Book a flight to a destination."""
    # Implementation
    return {"id": "FL123", "status": "confirmed"}

@tool
def cancel_flight(id: str) -> dict:
    """Cancel a flight booking."""
    # Implementation
    return {"status": "cancelled"}

@tool
def book_hotel(location: str, date: str) -> dict:
    """Book a hotel at a location."""
    # Implementation
    return {"booking_id": "HT456", "status": "confirmed"}

@tool
def cancel_hotel(booking_id: str) -> dict:
    """Cancel a hotel booking."""
    # Implementation
    return {"status": "cancelled"}

# Create compensation middleware
compensation = CompensationMiddleware(
    compensation_mapping={
        "book_flight": "cancel_flight",
        "book_hotel": "cancel_hotel",
    }
)

# Create agent with compensation
agent = create_agent(
    model="openai:gpt-4",
    tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
    middleware=[compensation],  # Add middleware here
)

# Use the agent normally - compensation happens automatically
result = agent.invoke({
    "messages": [{"role": "user", "content": "Book a flight and hotel for Paris"}]
})
```})
```

### Custom Parameter Mapping

When the compensation tool requires different parameters than what's in the result:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import CompensationMiddleware
from langchain_core.tools import tool

@tool
def book_flight(destination: str, date: str) -> dict:
    """Book a flight to destination."""
    return {
        "confirmation_number": "ABC123",
        "destination": destination,
        "date": date
    }

@tool
def cancel_flight(confirmation_number: str) -> str:
    """Cancel a flight booking using confirmation number."""
    return f"Flight {confirmation_number} cancelled"

# Custom mapper to extract confirmation number
def map_flight_booking(result, original_params):
    """Extract confirmation number from booking result."""
    return {"confirmation_number": result["confirmation_number"]}

# Create middleware with custom mappers
compensation = CompensationMiddleware(
    compensation_mapping={
        "book_flight": "cancel_flight",
    },
    state_mappers={
        "book_flight": map_flight_booking,
    }
)

agent = create_agent(
    model="openai:gpt-4",
    tools=[book_flight, cancel_flight],
    middleware=[compensation],
)
```

### Automatic Parameter Detection

If your result contains common ID field names, parameter mapping happens automatically:

```python
@tool
def create_resource(name: str) -> dict:
    """Create a cloud resource."""
    return {
        "id": "res-12345",  # Automatically detected!
        "name": name,
        "status": "created"
    }

@tool
def delete_resource(id: str) -> str:
    """Delete a cloud resource by ID."""
    return f"Resource {id} deleted"

# No state_mappers needed - 'id' is automatically detected
compensation = CompensationMiddleware(
    compensation_mapping={
        "create_resource": "delete_resource",
    }
)
```

Common auto-detected field names:
- `id`
- `booking_id`
- `resource_id`
- `transaction_id`
- `reservation_id`
- `order_id`
- Any field ending with `_id`

### Complex Example: Multi-Step Resource Allocation

```python
from langchain.agents import create_agent
from langchain.agents.middleware import CompensationMiddleware
from langchain_core.tools import tool

# Forward operations
@tool
def allocate_vm(specs: str) -> dict:
    """Allocate a virtual machine."""
    return {"vm_id": "vm-001", "ip": "10.0.0.1"}

@tool
def configure_networking(vm_id: str) -> dict:
    """Configure networking for a VM."""
    return {"network_id": "net-001", "vm_id": vm_id}

@tool
def install_software(vm_id: str, software: str) -> dict:
    """Install software on a VM."""
    # This might fail
    if software_unavailable(software):
        raise ValueError(f"Software {software} not available")
    return {"vm_id": vm_id, "installed": software}

# Compensation operations
@tool
def deallocate_vm(vm_id: str) -> str:
    """Deallocate a virtual machine."""
    return f"VM {vm_id} deallocated"

@tool
def remove_networking(network_id: str) -> str:
    """Remove network configuration."""
    return f"Network {network_id} removed"

# Custom mappers
def map_vm_allocation(result, params):
    return {"vm_id": result["vm_id"]}

def map_networking(result, params):
    return {"network_id": result["network_id"]}

# Create compensation middleware
compensation = CompensationMiddleware(
    compensation_mapping={
        "allocate_vm": "deallocate_vm",
        "configure_networking": "remove_networking",
        # install_software has no compensation - if it fails, previous steps rollback
    },
    state_mappers={
        "allocate_vm": map_vm_allocation,
        "configure_networking": map_networking,
    }
)

agent = create_agent(
    model="openai:gpt-4",
    tools=[
        allocate_vm, configure_networking, install_software,
        deallocate_vm, remove_networking
    ],
    middleware=[compensation],
)

# If install_software fails, automatically:
# 1. remove_networking (most recent successful compensatable action)
# 2. deallocate_vm (second most recent)
```

## Advanced Features

### Error Detection

The middleware automatically detects failures by checking:

1. **ToolMessage status**: If `status == "error"`
2. **Content inspection**: Looking for error indicators like:
   - `"Error:"` in content
   - `"error"` in lowercase content
   - `"failed"` or `"exception"` in content

You can customize error detection by extending the middleware.

### State Tracking

The middleware automatically extends the agent state with:

```python
{
    "compensation_stack": CompensationStack(),  # Tracks compensatable actions
    "compensated_actions": [],  # List of actions that were compensated
}
```

Access compensation history:

```python
result = agent.invoke({"messages": [...]})

# Check if any compensation occurred
if result.get("compensated_actions"):
    print("Compensation was performed:")
    for action in result["compensated_actions"]:
        print(f"  - {action['original_tool']} compensated by {action['compensation_tool']}")
```

### Combining with Other Middleware

Compensation middleware works seamlessly with other middleware:

```python
from langchain.agents.middleware import (
    CompensationMiddleware,
    ToolRetryMiddleware,
    HumanInTheLoopMiddleware,
)

agent = create_agent(
    model="openai:gpt-4",
    tools=[...],
    middleware=[
        ToolRetryMiddleware(max_retries=2),  # Retry before giving up
        CompensationMiddleware(compensation_mapping={...}),  # Compensate on final failure
        HumanInTheLoopMiddleware(),  # Human review if needed
    ],
)
```

**Execution Order**: Middleware is applied in the order specified, so retries happen before compensation.

## Benefits of Middleware-Based Approach

### 1. **Transparent to End Users**

The graph structure remains unchanged:

```python
# Same graph whether compensation is enabled or not
agent_without_comp = create_agent(model, tools, middleware=[])
agent_with_comp = create_agent(model, tools, middleware=[CompensationMiddleware(...)])

# Both have identical node structure: START → model → tools → model → END
```

### 2. **Research-Friendly**

Easy to A/B test compensation strategies:

```python
# Control group
control_agent = create_agent(model, tools, middleware=[])

# Treatment group
treatment_agent = create_agent(
    model, tools,
    middleware=[CompensationMiddleware(compensation_mapping=...)]
)

# Same API, same graph structure, different behavior
```

### 3. **Backward Compatible**

Existing agents continue to work without any changes. Compensation is opt-in via middleware.

### 4. **Composable**

Stack with other middleware for powerful combinations:

- Retry + Compensation: Try multiple times, then rollback on final failure
- Logging + Compensation: Track both attempts and rollbacks
- HITL + Compensation: Get human approval before compensating

### 5. **No Performance Overhead When Disabled**

If you don't use `CompensationMiddleware`, there's zero performance impact. The graph structure is identical to agents without compensation support.

## API Reference

### CompensationMiddleware

```python
class CompensationMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        compensation_mapping: Dict[str, str],
        state_mappers: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] | None = None,
        name: str = "compensation",
    ):
        """Initialize compensation middleware.

        Args:
            compensation_mapping: Maps tool names to their compensation tools.
                Example: {"book_flight": "cancel_flight"}
            state_mappers: Optional custom parameter mapping functions.
                Example: {"book_flight": lambda result, params: {"id": result["booking_id"]}}
            name: Middleware instance name (default: "compensation")
        """
```

**Methods**:

- `wrap_tool_call(request, handler)`: Sync tool execution wrapper
- `awrap_tool_call(request, handler)`: Async tool execution wrapper

### CompensationStack

```python
class CompensationStack:
    def push(self, record: CompensationRecord) -> None:
        """Add a compensatable action to the stack."""

    def pop(self) -> CompensationRecord | None:
        """Remove and return the most recent uncompensated action."""

    def mark_compensated(self, record_id: str) -> None:
        """Mark an action as compensated."""

    def get_uncompensated(self) -> list[CompensationRecord]:
        """Get all uncompensated actions in LIFO order."""

    def clear(self) -> None:
        """Clear the entire stack."""

    def __len__(self) -> int:
        """Return the number of uncompensated actions."""
```

### CompensationRecord

```python
class CompensationRecord(dict):
    """Record of a compensatable action."""

    id: str  # Unique identifier
    tool_name: str  # Name of the tool executed
    params: Dict[str, Any]  # Original tool parameters
    result: Any  # Tool execution result
    timestamp: float  # When the action occurred
    compensated: bool  # Whether it's been compensated
    compensation_tool: str | None  # Name of compensation tool
    metadata: Dict[str, Any]  # Additional metadata
```

## Migration Guide

### From Node-Based Compensation

If you were using the previous node-based approach (with `enable_compensation` parameter):

**Before** (Node-Based):
```python
agent = create_agent(
    model="openai:gpt-4",
    tools=[book_flight, cancel_flight],
    enable_compensation=True,
    compensation_mapping={"book_flight": "cancel_flight"},
    state_mappers={"book_flight": my_mapper},
)
```

**After** (Middleware-Based):
```python
from langchain.agents.middleware import CompensationMiddleware

compensation = CompensationMiddleware(
    compensation_mapping={"book_flight": "cancel_flight"},
    state_mappers={"book_flight": my_mapper},
)

agent = create_agent(
    model="openai:gpt-4",
    tools=[book_flight, cancel_flight],
    middleware=[compensation],  # Add as middleware
)
```

**Key Changes**:
1. Import `CompensationMiddleware` from `langchain.agents.middleware`
2. Create middleware instance with same configuration
3. Pass to `create_agent` via `middleware` parameter
4. Remove `enable_compensation`, `compensation_mapping`, and `state_mappers` from `create_agent` call

**Benefits**:
- Same functionality, cleaner API
- No extra graph nodes
- Easier to combine with other middleware
- More transparent to end users

## Best Practices

### 1. **Design Idempotent Compensation Tools**

Ensure compensation tools can be called multiple times safely:

```python
@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel a booking (idempotent)."""
    booking = get_booking(booking_id)
    if booking and booking.status != "cancelled":
        booking.cancel()
    return f"Booking {booking_id} cancelled"
```

### 2. **Return Structured Data**

Make compensation easier by returning dicts with clear IDs:

```python
# Good: Returns dict with clear identifier
@tool
def create_resource(name: str) -> dict:
    return {"id": "res-123", "name": name, "status": "created"}

# Less ideal: Returns string (harder to parse)
@tool
def create_resource(name: str) -> str:
    return "Created resource res-123"
```

### 3. **Handle Partial Failures**

Not all operations need compensation. Only map tools with meaningful side effects:

```python
compensation = CompensationMiddleware(
    compensation_mapping={
        "charge_credit_card": "refund_credit_card",  # Has side effect - needs compensation
        "send_email": "send_apology_email",  # Has side effect - needs compensation
        # "validate_input" - No side effect, no compensation needed
        # "check_availability" - Read-only, no compensation needed
    }
)
```

### 4. **Test Compensation Logic**

Write tests that simulate failures:

```python
def test_compensation_on_failure():
    """Test that compensation happens when a tool fails."""
    # Create test tools that track invocations
    invocations = []

    @tool
    def action1() -> dict:
        invocations.append("action1")
        return {"id": "1"}

    @tool
    def action2() -> dict:
        invocations.append("action2")
        raise ValueError("Simulated failure")

    @tool
    def undo_action1(id: str) -> str:
        invocations.append("undo_action1")
        return "undone"

    compensation = CompensationMiddleware(
        compensation_mapping={"action1": "undo_action1"}
    )

    agent = create_agent(
        model="openai:gpt-4",
        tools=[action1, action2, undo_action1],
        middleware=[compensation],
    )

    # Execute agent - action2 will fail
    result = agent.invoke({"messages": [...]})

    # Verify compensation occurred
    assert "undo_action1" in invocations
    assert invocations == ["action1", "action2", "undo_action1"]
```

### 5. **Monitor Compensation Events**

Track how often compensation occurs in production:

```python
import logging

class MonitoredCompensationMiddleware(CompensationMiddleware):
    def wrap_tool_call(self, request, handler):
        result = super().wrap_tool_call(request, handler)

        # Log compensation events
        if request.state.get("compensated_actions"):
            logging.warning(
                f"Compensation triggered after {request.tool_call['name']} failure"
            )

        return result
```

## Troubleshooting

### Compensation Not Triggering

**Problem**: Tools fail but compensation doesn't happen.

**Solutions**:
1. Check that forward tool is in `compensation_mapping`
2. Verify compensation tool is in the `tools` list
3. Ensure forward tool completed successfully before failure
4. Check error detection - does the error message contain error indicators?

### Wrong Parameters Passed to Compensation Tool

**Problem**: Compensation tool receives incorrect or missing parameters.

**Solutions**:
1. Provide a custom `state_mapper` function
2. Ensure forward tool returns a dict with clear ID fields
3. Check compensation tool parameter names match common ID patterns
4. Add logging to see what parameters are being extracted

### Compensation Order Issues

**Problem**: Compensation happens in the wrong order.

**Solution**: The middleware uses LIFO order automatically. If dependencies require different ordering, you may need to adjust the compensation mapping or use custom logic.

## Frequently Asked Questions

### Q: Does this add extra nodes to my graph?

**A**: No! The middleware-based approach doesn't modify the graph structure. Compensation happens transparently within the tool execution layer.

### Q: Can I use this with streaming?

**A**: Yes, compensation works with both streaming and non-streaming agents. The compensation logic executes after tool completion.

### Q: What happens if a compensation tool fails?

**A**: The middleware logs the failure and continues attempting to compensate remaining actions. Partial compensation is better than no compensation.

### Q: Can I inspect the compensation stack?

**A**: Yes, access it from the state:

```python
result = agent.invoke({"messages": [...]})
stack = result.get("compensation_stack")
if stack:
    uncompensated = stack.get_uncompensated()
    print(f"{len(uncompensated)} actions still uncompensated")
```

### Q: Does this work with async agents?

**A**: Yes! The middleware provides both `wrap_tool_call` (sync) and `awrap_tool_call` (async) implementations.

### Q: Can I disable compensation for specific invocations?

**A**: Yes, simply don't include the `CompensationMiddleware` when creating that agent instance, or create two separate agents.

## Related Documentation

- [Middleware Guide](https://docs.langchain.com/oss/python/langchain/middleware)
- [Agent Creation](https://docs.langchain.com/oss/python/langchain/agents)
- [Tool Development](https://docs.langchain.com/oss/python/langchain/tools)

## Conclusion

The `CompensationMiddleware` provides a powerful, transparent way to handle transaction-like rollbacks in agent workflows. By leveraging the middleware architecture, it remains completely invisible to end users while ensuring data consistency and error recovery.

Key takeaways:
- ✅ No graph structure changes
- ✅ LIFO rollback order
- ✅ Automatic parameter detection
- ✅ Composable with other middleware
- ✅ Zero overhead when not used

For questions or issues, please refer to the LangChain documentation or open an issue on GitHub.
    return {"booking_id": make_hotel_booking(location, date)}

@tool
def cancel_hotel(booking_id: str) -> str:
    """Cancel a hotel booking."""
    cancel_hotel_api_call(booking_id)
    return f"Hotel booking {booking_id} cancelled"

# Create agent with compensation
agent = create_agent(
    model="openai:gpt-4o",
    tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
    system_prompt="You are a travel booking assistant.",
    enable_compensation=True,
    compensation_mapping={
        "book_flight": "cancel_flight",
        "book_hotel": "cancel_hotel",
    },
    state_mappers={
        "book_flight": lambda result, params: {"booking_id": result["booking_id"]},
        "book_hotel": lambda result, params: {"booking_id": result["booking_id"]},
    },
)

# Run the agent
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Book me a flight to Paris and a hotel for tomorrow"}
    ]
})
```

**Behavior**:
- If `book_flight` succeeds but `book_hotel` fails → `cancel_flight` is automatically called
- The model receives a message about the failure and compensation
- The model can then suggest alternatives or report the issue to the user

### Advanced Example: Multi-Step Resource Allocation

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def allocate_compute(instance_type: str) -> dict:
    """Allocate compute resources."""
    instance_id = create_vm_instance(instance_type)
    return {"instance_id": instance_id, "type": instance_type}

@tool
def deallocate_compute(instance_id: str) -> str:
    """Deallocate compute resources."""
    terminate_vm_instance(instance_id)
    return f"Instance {instance_id} deallocated"

@tool
def allocate_storage(size_gb: int) -> dict:
    """Allocate storage volume."""
    volume_id = create_storage_volume(size_gb)
    return {"volume_id": volume_id, "size": size_gb}

@tool
def deallocate_storage(volume_id: str) -> str:
    """Deallocate storage volume."""
    delete_storage_volume(volume_id)
    return f"Volume {volume_id} deallocated"

@tool
def attach_storage(instance_id: str, volume_id: str) -> dict:
    """Attach storage to compute instance."""
    # This might fail if instance or volume not ready
    attach_result = attach_volume_to_instance(instance_id, volume_id)
    if not attach_result.success:
        raise RuntimeError("Failed to attach storage")
    return {"attachment_id": attach_result.id}

@tool
def detach_storage(attachment_id: str) -> str:
    """Detach storage from instance."""
    detach_volume(attachment_id)
    return f"Storage detached: {attachment_id}"

# Custom state mappers for complex parameter extraction
def map_compute_allocation(result, params):
    """Extract instance_id for deallocation."""
    return {"instance_id": result["instance_id"]}

def map_storage_allocation(result, params):
    """Extract volume_id for deallocation."""
    return {"volume_id": result["volume_id"]}

def map_attachment(result, params):
    """Extract attachment_id for detachment."""
    return {"attachment_id": result["attachment_id"]}

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[
        allocate_compute, deallocate_compute,
        allocate_storage, deallocate_storage,
        attach_storage, detach_storage,
    ],
    system_prompt="You are a cloud infrastructure provisioning assistant.",
    enable_compensation=True,
    compensation_mapping={
        "allocate_compute": "deallocate_compute",
        "allocate_storage": "deallocate_storage",
        "attach_storage": "detach_storage",
    },
    state_mappers={
        "allocate_compute": map_compute_allocation,
        "allocate_storage": map_storage_allocation,
        "attach_storage": map_attachment,
    },
    debug=True,  # Enable to see compensation in action
)
```

**Execution Flow**:

1. User: "Set up a server with 100GB storage"
2. Agent calls `allocate_compute` → Success (instance_id: "i-123")
3. Agent calls `allocate_storage` → Success (volume_id: "vol-456")
4. Agent calls `attach_storage` → **Fails** (instance not ready)
5. **Compensation triggered** (LIFO order):
   - Call `deallocate_storage(volume_id="vol-456")`
   - Call `deallocate_compute(instance_id="i-123")`
6. Agent receives recovery message and can retry or report failure

### Example Without State Mappers (Using Heuristics)

```python
@tool
def create_database(name: str) -> dict:
    """Create a database."""
    db_id = create_db(name)
    return {"id": db_id, "name": name}  # Note: using 'id' field

@tool
def delete_database(id: str) -> str:  # Parameter name matches result field
    """Delete a database."""
    delete_db(id)
    return f"Database {id} deleted"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[create_database, delete_database],
    enable_compensation=True,
    compensation_mapping={
        "create_database": "delete_database",
    },
    # No state_mappers needed - system will find 'id' field automatically
)
```

The compensation system will automatically extract `id` from the result and pass it to `delete_database`.

## Implementation Details

### Error Detection Logic

The `error_detection` node examines `ToolMessage` objects to determine if a failure occurred:

```python
def is_error_message(tool_message: ToolMessage) -> bool:
    content_str = str(tool_message.content)
    return "Error:" in content_str or "error" in content_str.lower()
```

**Tracked as Success** (recorded for compensation):
- Tool execution completed without exception
- ToolMessage content does NOT contain "Error:" or "error"
- Tool is listed in `compensation_mapping`

**Triggered as Failure**:
- ToolMessage content contains "Error:" or "error" (case-insensitive)
- Tool raised an exception (caught by ToolNode)

### Parameter Mapping Heuristics

When no `state_mapper` is provided, the system uses this fallback logic:

1. **Check common ID fields**: Searches result dict for `id`, `booking_id`, `resource_id`, `transaction_id`, `reservation_id`
2. **Match compensation tool parameters**: If compensation tool has `args_schema`, extract fields from result that match parameter names
3. **No match found**: Skip compensation and log warning

### Tool Result Extraction

Results are extracted from `ToolMessage.content`:

```python
def _extract_tool_result_from_message(msg: ToolMessage) -> Any:
    content = msg.content

    # Already a dict
    if isinstance(content, dict):
        return content

    # Try to parse JSON string
    if isinstance(content, str) and content.startswith("{") and content.endswith("}"):
        try:
            return json.loads(content)
        except:
            return content

    return content
```

### Recovery Mode Behavior

After compensation completes:
1. `recovery_mode` is set to `True` in state
2. Next model invocation receives additional context:
   ```
   "Previous action failed. Compensation has been performed.
    Consider alternative approaches or report the failure."
   ```
3. Model can:
   - Suggest alternative approaches
   - Ask for user input
   - Report the failure gracefully
   - Retry with different parameters

## Limitations and Considerations

### 1. Tool Requirements

**Both forward and compensation tools must be in the `tools` list**:
```python
# ✅ Correct
tools = [book_flight, cancel_flight]

# ❌ Incorrect - cancel_flight not in tools
tools = [book_flight]
compensation_mapping = {"book_flight": "cancel_flight"}
```

### 2. Provider (Built-in) Tools Not Supported

Compensation only works with **client-side tools** (BaseTool instances):
- ✅ Decorated with `@tool`
- ✅ Subclass of `BaseTool`
- ❌ Provider-native tools (dict format)

### 3. Idempotency Considerations

Compensation tools should be **idempotent** when possible:
```python
@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel a booking (idempotent)."""
    if not booking_exists(booking_id):
        return f"Booking {booking_id} already cancelled"
    cancel_booking_api(booking_id)
    return f"Booking {booking_id} cancelled"
```

### 4. Compensation May Fail

If compensation itself fails:
- Error is logged in the message stream as `AIMessage`
- `recovery_mode` is still set to `True`
- Execution continues (does not halt the agent)

### 5. No Nested Compensation

If a compensation action itself triggers a failure:
- Further compensation is **not** triggered
- This prevents infinite compensation loops

### 6. State Persistence

When using `checkpointer`:
- `CompensationStack` is included in state
- Can resume from checkpoint with compensation history intact
- Useful for long-running agents with multiple retry attempts

## Testing Compensation

### Unit Test Example

```python
import pytest
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def allocate_resource(resource_type: str) -> dict:
    """Allocate a resource."""
    return {"resource_id": f"{resource_type}-001"}

@tool
def deallocate_resource(resource_id: str) -> str:
    """Deallocate a resource."""
    return f"Deallocated {resource_id}"

@tool
def configure_resource(resource_id: str) -> dict:
    """Configure resource - simulates failure."""
    raise ValueError("Configuration failed")

def test_compensation_triggered():
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[allocate_resource, deallocate_resource, configure_resource],
        enable_compensation=True,
        compensation_mapping={
            "allocate_resource": "deallocate_resource",
        },
        state_mappers={
            "allocate_resource": lambda r, p: {"resource_id": r["resource_id"]},
        },
    )

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Allocate and configure a compute resource"}
        ]
    })

    messages = result["messages"]

    # Verify compensation happened
    assert any("Deallocated" in str(m.content) for m in messages)
    # Verify recovery mode message
    assert any("Compensation has been performed" in str(m.content) for m in messages)
```

## Migration from `create_react_agent`

If you're using the legacy `create_react_agent` with compensation:

**Before** (legacy):
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="openai:gpt-4",
    tools=[...],
    enable_compensation=True,
    compensation_mapping={...},
    state_mappers={...},
    version="v1",  # Required for compensation
)
```

**After** (new):
```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4",
    tools=[...],
    enable_compensation=True,
    compensation_mapping={...},
    state_mappers={...},
    # No version parameter needed - works with default architecture
)
```

**Key Differences**:
1. `create_react_agent` required `version="v1"` for compensation
2. `create_agent` supports compensation without version restrictions
3. `create_agent` integrates compensation with middleware architecture
4. State schema is automatically extended (no manual schema definition needed)

## Best Practices

### 1. Design Tools with Compensation in Mind

```python
# ✅ Good: Returns structured data with IDs
@tool
def create_order(items: list[str]) -> dict:
    order_id = place_order(items)
    return {"order_id": order_id, "items": items, "status": "created"}

# ❌ Less ideal: Returns unstructured string
@tool
def create_order(items: list[str]) -> str:
    order_id = place_order(items)
    return f"Order {order_id} created"
```

### 2. Provide Explicit State Mappers

Prefer explicit mappers over relying on heuristics:

```python
# ✅ Explicit and clear
state_mappers = {
    "create_order": lambda r, p: {"order_id": r["order_id"]},
}

# ⚠️ Relies on heuristics (may break if result structure changes)
# No state_mapper provided
```

### 3. Handle Partial Failures Gracefully

```python
@tool
def cancel_order(order_id: str) -> str:
    """Cancel order - handles already cancelled."""
    try:
        cancel_order_api(order_id)
        return f"Order {order_id} cancelled"
    except OrderNotFoundError:
        return f"Order {order_id} already cancelled or not found"
```

### 4. Use Descriptive Tool Names

```python
# ✅ Clear relationship
"provision_server" → "deprovision_server"
"acquire_lock" → "release_lock"

# ❌ Unclear
"setup_thing" → "cleanup"
```

### 5. Log Compensation Actions

Enable debug mode during development:

```python
agent = create_agent(
    ...,
    enable_compensation=True,
    debug=True,  # See compensation in action
)
```

## Troubleshooting

### Compensation Not Triggered

**Problem**: Tool fails but compensation doesn't run.

**Solutions**:
1. Verify `enable_compensation=True`
2. Check that tool name is in `compensation_mapping`
3. Ensure error message contains "Error:" or "error"
4. Verify compensation tool is in `tools` list

### Parameters Not Mapped Correctly

**Problem**: Compensation tool receives wrong parameters.

**Solutions**:
1. Provide explicit `state_mapper` for the tool
2. Ensure forward tool returns dict with ID fields
3. Check compensation tool parameter names match result fields
4. Debug by printing result structure

### Compensation Tool Not Found

**Problem**: Error message "Compensation tool X not found".

**Solutions**:
1. Add compensation tool to `tools` list
2. Verify tool name in `compensation_mapping` matches actual tool.name
3. Check for typos in tool names

## Performance Considerations

### Memory Usage

Each compensatable action adds a record to the stack:
- **Average size**: ~200-500 bytes per record
- **For 100 actions**: ~20-50 KB

### Execution Overhead

- **Error detection**: Minimal (inspects last few messages)
- **Compensation**: Depends on compensation tool latency
- **State updates**: Negligible

### Recommendations

- Use compensation for **critical operations** with side effects
- Consider **idempotent** operations to reduce compensation complexity
- Monitor compensation frequency in production

## Future Enhancements

Potential improvements for future versions:

1. **Nested Compensation**: Support compensation chains
2. **Async Compensation**: Parallel compensation of independent actions
3. **Compensation Policies**: Configurable retry strategies
4. **Compensation Logs**: Structured audit trail
5. **Partial Compensation**: Allow selective compensation
6. **Compensation Timeouts**: Handle slow compensation tools

## Summary

Compensation support in `create_agent` provides automatic rollback capabilities for agents performing operations with side effects. Key features:

✅ LIFO compensation order
✅ Automatic error detection
✅ Flexible parameter mapping
✅ Seamless middleware integration
✅ Recovery mode for LLM context
✅ Compatible with checkpointing

This feature is essential for building reliable agents that interact with external systems, databases, or APIs where partial failures must be handled gracefully.
