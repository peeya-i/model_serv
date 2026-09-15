import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time
from typing import Any, Optional
import uuid

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import uvicorn

# Default ports
DEFAULT_LISTEN_PORT = 8001
DEFAULT_MODEL_PORT = 8000

MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "your-internal-secure-gateway-token-xyz")
MODEL_NAME = os.environ.get("MODEL_NAME", "Llama-3.2-3B-Instruct")

# 1. Initialize FastAPI app and default model connection
initial_model_port = int(os.environ.get("MODEL_PORT", DEFAULT_MODEL_PORT))
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", f"http://127.0.0.1:{initial_model_port}/v1")

client = OpenAI(
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
    default_headers={"x-from-entity": "Agent", "x-to-entity": "Model"},
)
app = FastAPI(title="Local Agent Server")

# Import event_logger from project root and attach logging middleware
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import event_logger
app.middleware("http")(event_logger.create_logging_middleware("agent_server"))


# 2. Define Agent Tools
def calculate(expression: str) -> str:
    """Safely evaluates a basic mathematical expression."""
    try:
        # In production, use a safe AST parser
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


def lookup_system_status() -> str:
    """Returns local system status metrics."""
    return json.dumps({"status": "healthy", "gpu_vram": "5.6GB", "queue_depth": 0})


# Mapping of tool names to callable Python functions
AVAILABLE_TOOLS = {
    "calculate": calculate,
    "lookup_system_status": lookup_system_status,
}

# 3. Tool Schemas (OpenAI-compatible format)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic math calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '12 * 45'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_system_status",
            "description": "Get current system health and hardware status",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# 4. Request / Response schemas
class AgentQueryRequest(BaseModel):
    prompt: str
    session_id: str | None = "default"
    temperature: float | None = None


class AgentQueryResponse(BaseModel):
    reply: str
    tools_used: list[str]


def get_active_model_name() -> str:
    """Detects active model loaded in local model server, or falls back to MODEL_NAME."""
    global MODEL_NAME
    try:
        models = client.models.list()
        if models and models.data and len(models.data) > 0:
            return models.data[0].id
    except Exception:
        pass
    return MODEL_NAME


@app.get("/health")
def health_check():
    active_model = get_active_model_name()
    return {
        "status": "healthy",
        "service": "local_agent_server",
        "model": active_model,
        "model_base_url": str(client.base_url),
    }


# 5. Agent Reasoning & Execution Loop
@app.post("/agent/chat", response_model=AgentQueryResponse)
def run_agent(req: AgentQueryRequest):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to local tools. Use tools when needed.",
        },
        {"role": "user", "content": req.prompt},
    ]
    tools_executed = []
    active_model = get_active_model_name()
    supports_tools = "llama-3.2" in active_model.lower()
    tool_temp = 0.0 if req.temperature is None else req.temperature
    gen_temp = 0.7 if req.temperature is None else req.temperature

    try:
        # Step A: Query local model (with tools if supported by active model like Llama-3.2)
        if supports_tools:
            try:
                response = client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=tool_temp,
                    max_tokens=512,
                )
            except Exception as exc:
                err_str = str(exc)
                if "tool" in err_str.lower() and ("tool_choice" in err_str.lower() or "tool-call-parser" in err_str.lower() or "400" in err_str):
                    response = client.chat.completions.create(
                        model=active_model,
                        messages=messages,
                        temperature=gen_temp,
                        max_tokens=512,
                    )
                else:
                    raise exc
        else:
            response = client.chat.completions.create(
                model=active_model,
                messages=messages,
                temperature=gen_temp,
                max_tokens=512,
            )

        response_message = response.choices[0].message

        # Step B: If the model decided to call a tool
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")

                if func_name in AVAILABLE_TOOLS:
                    tools_executed.append(func_name)
                    # Log tool invocation
                    tool_req_id = f"tool-{uuid.uuid4().hex[:8]}"
                    tool_endpoint = f"/tool/{func_name}"
                    tool_url = f"http://127.0.0.1:{DEFAULT_LISTEN_PORT}{tool_endpoint}"
                    event_logger.log_event({
                        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
                        "request_id": tool_req_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "service": "tool",
                        "from_entity": "Agent",
                        "to_entity": f"Tool:{func_name}",
                        "type": "request",
                        "method": "INVOKE",
                        "endpoint": tool_endpoint,
                        "url": tool_url,
                        "headers": {"content-type": "application/json", "x-tool-name": func_name},
                        "payload": args,
                    })
                    t0_tool = time.time()
                    tool_output = AVAILABLE_TOOLS[func_name](**args)
                    tool_dur_ms = round((time.time() - t0_tool) * 1000, 2)
                    event_logger.log_event({
                        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
                        "request_id": tool_req_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "service": "tool",
                        "from_entity": f"Tool:{func_name}",
                        "to_entity": "Agent",
                        "type": "response",
                        "method": "INVOKE",
                        "status_code": 200,
                        "duration_ms": tool_dur_ms,
                        "endpoint": tool_endpoint,
                        "url": tool_url,
                        "headers": {"content-type": "application/json"},
                        "payload": {"result": tool_output},
                    })
                else:
                    tool_output = f"Error: Tool {func_name} not found"

                # Llama-3.2 / Qwen template requires single tool_call per assistant message
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_output),
                })

            # Step C: Generate final answer with tool observations
            final_response = client.chat.completions.create(
                model=active_model,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )
            return AgentQueryResponse(
                reply=final_response.choices[0].message.content or "",
                tools_used=tools_executed,
            )

        return AgentQueryResponse(
            reply=response_message.content or "",
            tools_used=[],
        )
    except Exception as e:
        return AgentQueryResponse(
            reply=f"Agent encountered error: {e}",
            tools_used=tools_executed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local Agent Server with Tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l", "--port",
        type=int,
        default=int(os.environ.get("PORT", DEFAULT_LISTEN_PORT)),
        help="Listening port for this agent server",
    )
    parser.add_argument(
        "-m", "--model-port",
        type=int,
        default=int(os.environ.get("MODEL_PORT", DEFAULT_MODEL_PORT)),
        help="Port where the local model server is running",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", None),
        help="Model name or path to query (defaults to auto-detection from model server)",
    )
    args = parser.parse_args()

    # Configure client to point to specified model server port
    target_model_url = f"http://127.0.0.1:{args.model_port}/v1"
    client = OpenAI(base_url=target_model_url, api_key=MODEL_API_KEY)
    if args.model:
        MODEL_NAME = args.model

    print(
        f"Starting Agent Server on http://127.0.0.1:{args.port} "
        f"(Target model: {target_model_url}, Default Model: {MODEL_NAME})"
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)
