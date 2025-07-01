import streamlit as st
import os
import requests, json
from datetime import datetime
from flytekit.configuration import Config, PlatformConfig
import union

# --------------------------- Sidebar inputs ---------------------------------
with st.sidebar:
    union_api_key = st.text_input("Union API Key", key="union_api_key", type="password")
    artifact_name = st.text_input(
        "Qwen Artifact URI",
        value="flyte://av0.2/chrismatteson/default/development/qwen-2_5-0_5B-Instruct@ap6cbd7l8dvgz965xfl4/n0/0/o0",
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_tokens = st.number_input("Max tokens", min_value=16, max_value=2048, value=512)
    qwen_endpoint = st.text_input(
        "qwen-service endpoint",
        value=os.getenv("QWEN_ENDPOINT", "https://summer-glade-f277a.apps.serverless-1.us-east-2.s.union.ai"),
    )
    st.markdown("[Union docs](https://www.union.ai)")

    st.divider()
    st.subheader("Union control plane")
    union_endpoint = st.text_input(
        "Union API endpoint",
        value=os.getenv("UNION_ENDPOINT", "https://serverless.union.ai"),
        help="https://<host> part of the URL in the Union UI.",
    )
    union_project = st.text_input("Project", value=os.getenv("UNION_PROJECT", "default"))
    union_domain = st.text_input("Domain", value=os.getenv("UNION_DOMAIN", "development"))

# --------------------------- Service Status Indicator -----------------------

# Autorefresh sidebar status every 5 s
from streamlit_autorefresh import st_autorefresh  # type: ignore

st_autorefresh(interval=5000, key="qwen_status_tick")

# Re-use the same placeholder across reruns
if "_qwen_status_box_main" not in st.session_state:
    st.session_state["_qwen_status_box_main"] = st.sidebar.empty()
status_box = st.session_state["_qwen_status_box_main"]

def _render_status(msg: str, level: str) -> None:
    """Render the status message only if it changed to avoid flashing."""

    prev = st.session_state.get("_qwen_status_prev")
    if msg == prev:
        return

    status_box.empty()
    if level == "success":
        status_box.success(msg)
    else:
        status_box.error(msg)
    st.session_state["_qwen_status_prev"] = msg

def update_status() -> bool:
    """Poll the /v1/models endpoint and refresh sidebar indicator.

    Returns True if the service is ready (HTTP 200 & JSON), else False.
    """

    if not qwen_endpoint:
        _render_status("qwen-service: ❓ endpoint not set", "error")
        return False

    try:
        r = requests.get(f"{qwen_endpoint}/v1/models", timeout=3)
        ok_json = r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json")

        if ok_json:
            _render_status("qwen-service: ✅ ready", "success")
            return True

        _render_status(f"qwen-service: 🔴 {r.status_code}", "error")
    except Exception:
        _render_status("qwen-service: 🔴 unreachable", "error")

    return False

service_ready = update_status()

# --------------------------- Remote completion helper -----------------------

def remote_completion(endpoint: str, history: list[dict[str, str]], api_key: str | None = None) -> str | None:
    """Call the qwen-service OpenAI-compatible endpoint and return assistant reply."""

    if not endpoint:
        st.error("⚠️ Provide the qwen-service endpoint in the sidebar or QWEN_ENDPOINT env var.")
        return None

    payload = {
        "model": "qwen2",  # model_id used in VLLMApp
        "messages": history,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    try:
        r = requests.post(
            f"{endpoint}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        r.raise_for_status()
        # Ensure JSON parseable
        try:
            data = r.json()
        except ValueError:
            snippet = r.text[:200]
            st.error(
                f"❗️Non-JSON response ({r.status_code}). First 200 chars:\n```{snippet}```"
            )
            return None

        return data["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as e:
        st.error(
            f"❗️HTTP {e.response.status_code}:\n```{e.response.text[:400]}```"
        )
    except Exception as e:
        st.error(f"❗️Request failed: {type(e).__name__}: {e}")
    return None

# ---------------------------------------------------------------------------
# Union workflow helpers
# ---------------------------------------------------------------------------

# We need these definitions *before* the UI's Search button can reference
# `query_essays_remote`.


@st.cache_data(show_spinner=False, max_entries=5)
def _get_remote(endpoint: str, project: str, domain: str, api_key: str) -> union.UnionRemote:
    """Return a UnionRemote client using client-credentials auth."""
    cfg = Config(
        platform=PlatformConfig(
            endpoint=endpoint,
            auth_mode="client_credentials",
            client_id=project,  # key name
            client_credentials_secret=api_key.strip() if api_key else "",
            insecure=False,
        )
    )
    return union.UnionRemote(config=cfg, default_project=project, default_domain=domain)


def _get_launch_plan(remote: union.UnionRemote):
    """Fetch the latest *download-chunk-embed.query_essays* launch plan."""
    return remote.fetch_launch_plan(name="download-chunk-embed.query_essays")


def query_essays_remote(query: str, max_results: int, endpoint: str, project: str, domain: str, api_key: str) -> list[dict]:
    """Execute the launch plan synchronously and return the list-of-dict result."""
    remote = _get_remote(endpoint, project, domain, api_key)
    lp = _get_launch_plan(remote)
    exec = remote.execute(lp, inputs={"query": query, "max_results": max_results}, wait=True)
    # The workflow returns list[dict] as its sole output.
    return exec.outputs.get("o0") or []

# --------------------------- Query / Search page -----------------------------

st.header("Retrieve Paul-Graham essay chunks")

query_text = st.text_input("Query", key="search_query")
max_results = st.number_input("Max results", min_value=1, max_value=20, value=5)
max_distance = st.slider("Max distance", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

if st.button("Search", key="do_search") and query_text.strip():
    try:
        with st.spinner("Querying workflow …"):
            docs = query_essays_remote(
                query=query_text,
                max_results=int(max_results),
                endpoint=union_endpoint,
                project=union_project,
                domain=union_domain,
                api_key=union_api_key,
            )

        # docs is a list of dicts: { 'document': str, 'distance': float, 'id': str }
        filtered = [d for d in docs if d.get("distance", 1.0) <= max_distance]

        if not filtered:
            st.info("No chunks within distance threshold.")
        else:
            for d in filtered:
                st.markdown(f"**ID:** `{d.get('id','n/a')}` • **distance:** {d.get('distance',0):.3f}")
                st.write(d.get("document", ""))
                st.markdown("---")
    except Exception as e:
        st.error(f"Search failed: {e}")