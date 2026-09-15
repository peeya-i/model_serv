# Plan: Log All API Call Fields & Enhanced Event Log Details Modal

Log comprehensive HTTP API fields (request/response headers, full URLs, query parameters, client network details, HTTP protocol version, method, status code, duration, and body payloads) across all API endpoints, and display every field in a modern, structured detail pop-up modal when a user clicks any row in the **Event Logs** page.

## User Review Required

> [!NOTE]
> All existing events in `logs/events.json` will continue to be fully supported and rendered gracefully. Newly logged events will contain the full spectrum of HTTP and invocation fields: `headers`, `url`, `endpoint`, `query_params`, `client`, `http_version`, `method`, `status_code`, `duration_ms`, `payload`, `from_entity`, `to_entity`, `request_id`, `event_id`, and `timestamp`.

## Proposed Changes

### Backend Logging System

#### [MODIFY] [event_logger.py](file:///home/pi-net/Documents/agent_eng_labs/model_serv/event_logger.py)
- **Enrich Request Event Logging**:
  - Capture `headers`: full dictionary of HTTP headers (`dict(request.headers)`).
  - Capture `url`: complete URL string (`str(request.url)`).
  - Capture `query_params`: parsed query parameters (`dict(request.query_params)`).
  - Capture `client`: client host and port string (`f"{request.client.host}:{request.client.port}"` or host).
  - Capture `http_version`: protocol version (`request.scope.get("http_version", "1.1")`).
  - Capture `method`: HTTP method (`request.method`).
- **Enrich Response Event Logging**:
  - Capture `headers`: full dictionary of response HTTP headers (`dict(response.headers)`).
  - Capture `url`: complete URL string (`str(request.url)`).
  - Capture `method`: HTTP method corresponding to the request.
  - Capture `client`: client address.
  - Capture `http_version`: protocol version.
  - Ensure standard error (500) and streaming (SSE) response paths also log these complete fields.

#### [MODIFY] [agents/agent_server.py](file:///home/pi-net/Documents/agent_eng_labs/model_serv/agents/agent_server.py) & [agents/langchain_agent.py](file:///home/pi-net/Documents/agent_eng_labs/model_serv/agents/langchain_agent.py)
- Ensure tool invocation events capture complete execution context: `method: "INVOKE"`, `url: f"local://tool/{func_name}"`, `headers`, `status_code: 200`, and precise execution `duration_ms`.
- In `langchain_agent.py`, eliminate redundant manual logging for `/agent/chat` that previously created duplicate entries with missing fields, allowing the HTTP middleware to capture complete request and response events.

---

### Web App & Frontend Event Inspector

#### [MODIFY] [templates/index.html](file:///home/pi-net/Documents/agent_eng_labs/model_serv/templates/index.html)
- **Enhanced Modal Dialog Window**:
  - Enlarge modal container (`max-width: 920px`, polished glassmorphism, responsive scroll).
  - Modern modal header with badges for Service, Event Type (Request/Response), HTTP Method, Status Code, and close button.
  - **Structured Metadata Grid**:
    - **Identification**: Event ID (with 1-click copy), Request ID (with 1-click copy), Timestamp (formatted UTC & local).
    - **Network & Routing**: From & To Entity badges, Route / Endpoint, Full URL, Client IP/Port, HTTP Version.
    - **Performance**: Status Code with color badge (2xx green, 4xx/5xx red), Duration in milliseconds and seconds.
  - **Interactive Tabbed Detail Panels**:
    - **Tab 1: 📦 Payload / Body**: Syntax-highlighted formatted JSON viewer with "Copy Payload" button.
    - **Tab 2: 🏷️ HTTP Headers**: Formatted key-value table and JSON viewer displaying all request or response headers, with "Copy Headers" button.
    - **Tab 3: 🔍 Query Parameters**: Dynamic table for query strings if present.
    - **Tab 4: 📑 Raw Event JSON**: The complete raw JSON object containing every logged field, with a dedicated "Copy Full JSON" button.
  - **Modal Footer Actions**:
    - "📋 Copy Full Event JSON" button.
    - "📋 Copy Payload" button.
    - "Close" button and `Esc` key shortcut handling.

---

## Verification Plan

### Automated & CLI Tests
- Run Python test script to:
  1. Trigger an API call through the model server / agent server.
  2. Read the latest event record from `logs/events.json`.
  3. Validate that `headers`, `url`, `method`, `endpoint`, `client`, `http_version`, `payload`, `status_code`, and `duration_ms` are present and non-empty.

### Manual / Browser Verification
- Open the web console dashboard.
- Navigate to the **Event Logs** page.
- Click on any event row to open the detailed log modal popup.
- Verify:
  - All metadata fields (Event ID, Request ID, Timestamp, Service, From, To, Route, Full URL, Client, HTTP Version, Status, Duration) are clearly rendered.
  - The tabs (Payload, HTTP Headers, Query Parameters, Raw Event JSON) switch smoothly.
  - The copy buttons work and copy the expected JSON to clipboard.
