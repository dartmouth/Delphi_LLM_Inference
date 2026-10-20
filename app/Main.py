import asyncio
import atexit
import os
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_dartmouth.exceptions import InvalidKeyError, ModelNotFoundError
from langchain_dartmouth.llms import ChatDartmouth
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

st.set_page_config(page_title="Data Analytics Agent", page_icon="💬")

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden}
        .stAppDeployButton {visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

API_KEY = os.environ.get("DARTMOUTH_CHAT_API_KEY")

if not API_KEY:
    st.error(
        "DARTMOUTH_CHAT_API_KEY is not set. Add it to your `.env` file or "
        "environment before running the app."
    )
    st.stop()


ALLOWED_MODELS = {"gptoss120b", "qwen35122b"}

# Token cap per model (keyed like ALLOWED_MODELS). The Dartmouth model listing
# does not expose limits, so these are the models' published context windows.
MODEL_TOKEN_CAPS = {"gptoss120b": 131072, "qwen35122b": 262144}
DEFAULT_TOKEN_CAP = 32768

MCP_SERVER_PATH = Path(__file__).parent / "lib" / "mcp_stats.py"
SYSTEM_PROMPT = """You are a data analytics assistant for Delphi-study survey data.
The user may upload a CSV or Excel file; you can read it only through your tools.
Workflow for building the statistics table:
1. Call list_uploaded_files, then list_question_columns for the file.
2. Call build_statistics_table. Only pass options the user asked for (e.g.
   combine_groups to merge answer categories, show_count for counts).
3. Reply with the returned markdown table copied verbatim, followed by a short
   summary. Never invent numbers; if a tool returns an error, explain it and retry
   with corrected arguments.
Use compute_consensus when the user asks about consensus or percent agreement.
If no file has been uploaded, ask the user to upload one in the sidebar."""


def _session_upload_dir() -> Path:
    """Per-browser-session folder the MCP server is allowed to read from."""
    if "upload_dir" not in st.session_state:
        path = Path(tempfile.mkdtemp(prefix="agentic_uploads_"))
        atexit.register(shutil.rmtree, path, ignore_errors=True)
        st.session_state.upload_dir = str(path)
    return Path(st.session_state.upload_dir)


def _save_upload(uploaded) -> str:
    """Write the Streamlit upload into the session folder; return its safe file name."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(uploaded.name).name)
    (_session_upload_dir() / name).write_bytes(uploaded.getvalue())
    return name


def _tool_text(content) -> str:
    """MCP tool results arrive as a list of content blocks; join their text."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


@st.cache_resource(show_spinner=False)
def _get_event_loop() -> asyncio.AbstractEventLoop:
    """One long-lived loop shared by all reruns.

    The cached LLM holds async HTTP clients bound to the loop they first ran
    on; asyncio.run() closes its loop each time, breaking them on the next call.
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="agent-loop", daemon=True).start()
    return loop


def _run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _get_event_loop()).result()


async def _run_agent(llm, history, upload_dir: Path):
    client = MultiServerMCPClient(
        {
            "delphi-stats": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(MCP_SERVER_PATH)],
                "env": {"MCP_UPLOAD_DIR": str(upload_dir)},
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return await agent.ainvoke({"messages": history})


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@st.cache_data(ttl=3600, show_spinner="Loading available models...")
def load_models(api_key: str):
    models = ChatDartmouth.list(dartmouth_chat_api_key=api_key, base_only=True)
    models = [
        m
        for m in models
        if any(allowed in _normalize(m.name or m.id) for allowed in ALLOWED_MODELS)
    ]
    return sorted(models, key=lambda m: (m.name or m.id).lower())


@st.cache_resource(show_spinner=False)
def get_llm(
    model_id: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
):
    # Qwen is a reasoning model: it spends output tokens on hidden "thinking"
    # before writing the visible answer, so it needs a larger budget than the
    # library default (512) to avoid getting cut off before any answer is written.
    # Tool-calling turns also need room for the table the model copies back.
    # The limit is user-adjustable from the sidebar.
    # reasoning_effort=None ("Default") sends nothing, so the model's own default applies.
    model_kwargs = {}
    if reasoning_effort:
        model_kwargs["reasoning_effort"] = reasoning_effort
    return ChatDartmouth(
        model_name=model_id,
        dartmouth_chat_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
    )


try:
    models = load_models(API_KEY)
except InvalidKeyError:
    st.error("The configured DARTMOUTH_CHAT_API_KEY was rejected. Check your key.")
    st.stop()
except Exception as e:
    st.error(f"Could not load models from Dartmouth Chat: {e}")
    st.stop()

if not models:
    st.error("Dartmouth Chat did not return any available models.")
    st.stop()

st.title("💬 Data Analytics Agent")

with st.sidebar:
    st.header("Data file")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        st.caption(f"Saved as `{_save_upload(uploaded)}` for the analytics tools.")

    st.header("Model")
    labels = [f"{m.name or m.id}  ·  {m.cost or 'undefined'}" for m in models]
    default_index = next(
        (i for i, m in enumerate(models) if m.id == "openai.gpt-oss-120b"), 0
    )
    selected_index = st.selectbox(
        "Choose a model",
        options=range(len(models)),
        format_func=lambda i: labels[i],
        index=default_index,
    )
    selected_model = models[selected_index]

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    effort_choice = st.selectbox(
        "Reasoning effort",
        options=["Default", "Low", "Medium", "High"],
        index=2,
        help="Higher effort lets reasoning models think longer. It is slower and uses more output tokens.",
    )
    reasoning_effort = None if effort_choice == "Default" else effort_choice.lower()

    # Reasoning models (e.g. Qwen) need a larger default budget; the key makes
    # the input reset to the right default when the model changes.
    default_max_tokens = 4096 if "qwen" in _normalize(selected_model.id) else 2048
    model_cap = next(
        (
            cap
            for key, cap in MODEL_TOKEN_CAPS.items()
            if key in _normalize(selected_model.name or selected_model.id)
        ),
        DEFAULT_TOKEN_CAP,
    )
    max_tokens = st.number_input(
        f"Max output tokens (max:{model_cap})",
        min_value=256,
        max_value=model_cap,
        value=default_max_tokens,
        step=512,
        key=f"max_tokens_{selected_model.id}",
        help="Raise this if responses are cut off before finishing.",
    )

    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            llm = get_llm(
                selected_model.id,
                API_KEY,
                temperature,
                int(max_tokens),
                reasoning_effort,
            )
            history = [
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]
            with st.spinner("Analysing..."):
                result = _run_async(_run_agent(llm, history, _session_upload_dir()))
            response = result["messages"][-1]
            tool_outputs = [
                m for m in result["messages"][len(history):] if isinstance(m, ToolMessage)
            ]
            if tool_outputs:
                with st.expander("Tool calls"):
                    for m in tool_outputs:
                        st.markdown(f"**{m.name}**")
                        st.markdown(_tool_text(m.content))
            if response.response_metadata.get("finish_reason") == "length":
                st.warning(
                    "The response was cut off because it hit the model's "
                    "token limit before finishing. Try a shorter prompt or "
                    "increase 'Max output tokens' in the sidebar."
                )
            if response.content:
                placeholder.markdown(response.content)
            else:
                placeholder.info("The model returned an empty response.")
            st.session_state.messages.append(
                {"role": "assistant", "content": response.content}
            )
        except ModelNotFoundError:
            placeholder.error(
                f"Model '{selected_model.id}' was not found. Try selecting a "
                "different model."
            )
        except InvalidKeyError:
            placeholder.error("The configured DARTMOUTH_CHAT_API_KEY was rejected.")
        except Exception as e:
            placeholder.error(f"Error calling Dartmouth Chat API: {e}")
