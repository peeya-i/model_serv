import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import requests

# 1. Default ports and configuration
DEFAULT_LISTEN_PORT = 8002
DEFAULT_MODEL_PORT = 8001
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "your-internal-secure-gateway-token-xyz")
MODEL_NAME = os.environ.get("MODEL_NAME", "Llama-3.2-3B-Instruct")

app = FastAPI(title="LangChain Local Server Agent")

# Attach logging middleware from event_logger
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import event_logger
app.middleware("http")(event_logger.create_logging_middleware("langchain_agent"))


# 2. Define Agent Tools
@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Safely calculates a basic mathematical expression, e.g. '125 * 8'."""
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


TOOLS = [get_current_time, calculate]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# 3. Request / Response schemas matching web console contract
class AgentQueryRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"


class AgentQueryResponse(BaseModel):
    reply: str
    tools_used: list[str]


# Global state for active model and connection
runtime_config = {
    "model_port": DEFAULT_MODEL_PORT,
    "model_name": MODEL_NAME,
    "listen_port": DEFAULT_LISTEN_PORT,
}


def get_active_model_name() -> str:
    """Detects active model dynamically from the model server, or falls back to configured model."""
    target_url = f"http://127.0.0.1:{runtime_config['model_port']}/v1/models"
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}
    try:
        resp = requests.get(target_url, headers=headers, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data") and len(data["data"]) > 0:
                return data["data"][0]["id"]
    except Exception:
        pass
    return runtime_config["model_name"]


def get_llm_instance(model_name: str, with_tools: bool = True):
    """Creates a ChatOpenAI LangChain instance pointing to the target local model server."""
    base_url = f"http://127.0.0.1:{runtime_config['model_port']}/v1"
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=MODEL_API_KEY,
        model=model_name,
        temperature=0.0,
        max_tokens=512,
    )
    if with_tools:
        return llm.bind_tools(TOOLS), llm
    return llm, llm


@app.get("/health")
def health_check():
    """Health check endpoint polled by web console."""
    active_model = get_active_model_name()
    base_url = f"http://127.0.0.1:{runtime_config['model_port']}/v1"
    return {
        "status": "healthy",
        "service": "langchain_agent",
        "model": active_model,
        "model_base_url": base_url,
    }


@app.post("/agent/chat", response_model=AgentQueryResponse)
def chat_endpoint(req: AgentQueryRequest):
    """API endpoint called by the Web App console to execute agent queries."""
    query = req.prompt.strip()
    if not query:
        return AgentQueryResponse(reply="Please provide a prompt.", tools_used=[])

    request_id = f"req-{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()
    tools_executed = []

    # Log incoming request event
    event_logger.log_event({
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "langchain_agent",
        "type": "request",
        "endpoint": "/agent/chat",
        "payload": {"query": query},
    })

    active_model = get_active_model_name()
    supports_tools = "llama-3.2" in active_model.lower()
    messages = [
        SystemMessage(content="You are a helpful assistant with access to local tools. Always call tools when needed."),
        HumanMessage(content=query),
    ]

    try:
        # Step A: Query model (with tools if supported by active model like Llama-3.2)
        if supports_tools:
            llm_bound, llm_plain = get_llm_instance(active_model, with_tools=True)
            try:
                ai_msg = llm_bound.invoke(messages)
            except Exception as exc:
                err_str = str(exc)
                if "tool" in err_str.lower() and ("tool_choice" in err_str.lower() or "tool-call-parser" in err_str.lower() or "400" in err_str):
                    ai_msg = llm_plain.invoke(messages)
                else:
                    raise exc
        else:
            _, llm_plain = get_llm_instance(active_model, with_tools=False)
            ai_msg = llm_plain.invoke(messages)

        # Step B: Execute tools if requested by the model
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args") or {}

                if tool_name in TOOLS_BY_NAME:
                    tools_executed.append(tool_name)
                    tool_output = TOOLS_BY_NAME[tool_name].invoke(tool_args)
                else:
                    tool_output = f"Error: Tool '{tool_name}' not found"

                messages.append(ai_msg)
                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call.get("id", f"call-{uuid.uuid4().hex[:8]}"),
                        name=tool_name,
                    )
                )

            # Generate final response with observations
            final_response = llm_plain.invoke(messages)
            reply_text = final_response.content
        else:
            reply_text = ai_msg.content

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        event_logger.log_event({
            "event_id": f"evt-{uuid.uuid4().hex[:12]}",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "langchain_agent",
            "type": "response",
            "status_code": 200,
            "duration_ms": duration_ms,
            "endpoint": "/agent/chat",
            "payload": {
                "reply": reply_text,
                "tools_executed": tools_executed,
            },
        })

        return AgentQueryResponse(reply=reply_text, tools_used=tools_executed)

    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        event_logger.log_event({
            "event_id": f"evt-{uuid.uuid4().hex[:12]}",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "langchain_agent",
            "type": "response",
            "status_code": 500,
            "duration_ms": duration_ms,
            "endpoint": "/agent/chat",
            "payload": {
                "error": str(exc),
                "tools_executed": tools_executed,
            },
        })
        # Graceful error reply rather than crashing with unhandled 500
        return AgentQueryResponse(
            reply=f"LangChain Agent encountered an error: {exc}",
            tools_used=tools_executed,
        )


def run_agent(query: str) -> str:
    """Direct programmatic entrypoint for testing."""
    req = AgentQueryRequest(prompt=query)
    res = chat_endpoint(req)
    return res.reply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangChain Local Server Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l", "--port",
        type=int,
        default=int(os.environ.get("PORT", DEFAULT_LISTEN_PORT)),
        help="Listening port for this LangChain agent server",
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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run direct test prompts without starting HTTP server",
    )
    args = parser.parse_args()

    runtime_config["listen_port"] = args.port
    runtime_config["model_port"] = args.model_port
    if args.model:
        runtime_config["model_name"] = args.model

    target_url = f"http://127.0.0.1:{args.model_port}/v1"

    if args.test:
        print("=== Testing LangChain Local Server Agent ===")
        print(run_agent("What time is it right now?"))
        print(run_agent("Calculate 125 * 8"))
    else:
        print(
            f"Starting LangChain Agent Server on http://127.0.0.1:{args.port} "
            f"(Target model server: {target_url}, Default Model: {runtime_config['model_name']})"
        )
        uvicorn.run(app, host="127.0.0.1", port=args.port)
