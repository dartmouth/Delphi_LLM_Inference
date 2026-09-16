import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_dartmouth.exceptions import InvalidKeyError, ModelNotFoundError
from langchain_dartmouth.llms import ChatDartmouth

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
def get_llm(model_id: str, api_key: str, temperature: float):
    # Qwen is a reasoning model: it spends output tokens on hidden "thinking"
    # before writing the visible answer, so it needs a larger budget than the
    # library default (512) to avoid getting cut off before any answer is written.
    max_tokens = 4096 if "qwen" in _normalize(model_id) else 512
    return ChatDartmouth(
        model_name=model_id,
        dartmouth_chat_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
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

    if selected_model.description:
        st.caption(selected_model.description)
    if selected_model.capabilities:
        st.caption("Capabilities: " + ", ".join(selected_model.capabilities))
    st.caption(f"Hosted {'on-premises' if selected_model.is_local else 'by a third party'}")

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

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
            llm = get_llm(selected_model.id, API_KEY, temperature)
            history = [
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]
            with st.spinner("Thinking..."):
                response = llm.invoke(history)
            if response.response_metadata.get("finish_reason") == "length":
                st.warning(
                    "The response was cut off because it hit the model's "
                    "token limit before finishing. Try a shorter prompt or "
                    "raise `max_tokens`."
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
