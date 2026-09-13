import fcntl
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

# Determine directory paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "events.json")

# Ensure logs directory exists on module load
os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("[]\n")


# Maximum log file size: 200MB default, configurable via environment variable
MAX_LOG_SIZE_BYTES = int(os.environ.get("MAX_LOG_SIZE_BYTES", 200 * 1024 * 1024))


def _serialize_events(events: list[Any]) -> str:
    """Serializes the events list to formatted JSON string."""
    return json.dumps(events, indent=2, ensure_ascii=False, default=str) + "\n"


def log_event(event_data: dict[str, Any]) -> None:
    """Thread-safe and process-safe appender for events.json using file locking.
    
    Guarantees the log file never exceeds MAX_LOG_SIZE_BYTES (200MB) by purging
    the oldest log events from the beginning of the array.
    """
    # Do not log health checks, pings, or internal status polls
    endpoint = str(event_data.get("endpoint", "")).lower()
    if endpoint in {"/health", "/ping", "/healthz", "/v1/models"} or endpoint.startswith(("/health", "/ping")):
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read().strip()
            events: list[Any] = []
            if content:
                try:
                    loaded = json.loads(content)
                    if isinstance(loaded, list):
                        events = loaded
                    else:
                        events = [loaded]
                except Exception:
                    events = []
            
            events.append(event_data)

            # Serialize and check size against limit
            serialized = _serialize_events(events)
            serialized_bytes = serialized.encode("utf-8")

            # Purge older events from the front if the file exceeds MAX_LOG_SIZE_BYTES
            while events and len(serialized_bytes) > MAX_LOG_SIZE_BYTES:
                excess = len(serialized_bytes) - MAX_LOG_SIZE_BYTES
                avg_item_size = max(1, len(serialized_bytes) // len(events))
                items_to_remove = max(1, (excess // avg_item_size) + 1)
                events = events[items_to_remove:]
                serialized = _serialize_events(events)
                serialized_bytes = serialized.encode("utf-8")
            
            f.seek(0)
            f.truncate()
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def create_logging_middleware(service_name: str) -> Callable:
    """Creates a FastAPI/Starlette HTTP middleware that captures complete request and response payloads."""
    from starlette.concurrency import iterate_in_threadpool
    from starlette.requests import Request
    from starlette.responses import Response

    async def logging_middleware(request: Request, call_next: Callable) -> Response:
        # Avoid double-logging if middleware is attached multiple times
        if getattr(request.state, f"_logged_by_{service_name}", False):
            return await call_next(request)
        setattr(request.state, f"_logged_by_{service_name}", True)

        # Ignore static assets, docs, health checks, and pings to keep events.json clean
        path = request.url.path
        if (
            path in {"/favicon.ico", "/docs", "/redoc", "/openapi.json", "/health", "/ping", "/healthz", "/v1/models"}
            or path.startswith(("/health", "/ping"))
        ):
            return await call_next(request)

        # Request ID tracking
        request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
        
        # Read and parse request payload
        req_body_bytes = await request.body()
        req_payload: Any = None
        if req_body_bytes:
            try:
                req_payload = json.loads(req_body_bytes.decode("utf-8"))
            except Exception:
                req_payload = req_body_bytes.decode("utf-8", errors="replace")
        elif request.query_params:
            req_payload = dict(request.query_params)
        else:
            req_payload = {}

        # Log incoming request
        log_event({
            "event_id": f"evt-{uuid.uuid4().hex[:12]}",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": service_name,
            "type": "request",
            "method": request.method,
            "endpoint": request.url.path,
            "payload": req_payload,
        })

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            log_event({
                "event_id": f"evt-{uuid.uuid4().hex[:12]}",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": service_name,
                "type": "response",
                "status_code": 500,
                "duration_ms": duration_ms,
                "endpoint": request.url.path,
                "payload": {"error": str(exc)},
            })
            raise exc

        content_type = response.headers.get("content-type", "")

        # Handle SSE / streaming responses
        if content_type.startswith("text/event-stream"):
            async def streamed_iterator():
                accumulated_chunks: list[bytes] = []
                try:
                    async for chunk in response.body_iterator:
                        accumulated_chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
                        yield chunk
                finally:
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    raw_text = b"".join(accumulated_chunks).decode("utf-8", errors="replace")
                    log_event({
                        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
                        "request_id": request_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "service": service_name,
                        "type": "response",
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                        "endpoint": request.url.path,
                        "payload": raw_text,
                    })
            response.body_iterator = streamed_iterator()
            return response

        # Handle standard non-streaming responses
        resp_body_chunks = [chunk async for chunk in response.body_iterator]
        response.body_iterator = iterate_in_threadpool(iter(resp_body_chunks))
        full_body = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8") for chunk in resp_body_chunks)

        resp_payload: Any = None
        if full_body:
            try:
                resp_payload = json.loads(full_body.decode("utf-8"))
            except Exception:
                resp_payload = full_body.decode("utf-8", errors="replace")
        else:
            resp_payload = {}

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_event({
            "event_id": f"evt-{uuid.uuid4().hex[:12]}",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": service_name,
            "type": "response",
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "endpoint": request.url.path,
            "payload": resp_payload,
        })

        return response

    return logging_middleware


# Default pre-configured middleware instance for vLLM
vllm_logging_middleware = create_logging_middleware("llm")
