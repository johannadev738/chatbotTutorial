import streamlit as st
from typing import Generator
from groq import Groq


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_icon="💬",
    layout="wide",
    page_title="LLM Chat Demo"
)


# --------------------------------------------------
# PAGE ICON
# --------------------------------------------------

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("💬")

st.subheader(
    "BubbleChat 0 to LLM Chatbot Demo Class",
    divider="rainbow",
    anchor=False
)


# --------------------------------------------------
# GROQ CLIENT
# --------------------------------------------------

try:
    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

except KeyError:
    st.error(
        "GROQ_API_KEY was not found in Streamlit Secrets."
    )
    st.stop()


# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None


# --------------------------------------------------
# AVAILABLE MODELS
# --------------------------------------------------

models = {
    "openai/gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "tokens": 8192,
        "developer": "OpenAI",
    },
    "openai/gpt-oss-120b": {
        "name": "GPT-OSS 120B",
        "tokens": 8192,
        "developer": "OpenAI",
    },
}


# --------------------------------------------------
# MODEL SETTINGS
# --------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=0
    )


# Clear chat when user switches models

if st.session_state.selected_model != model_option:

    st.session_state.messages = []

    st.session_state.selected_model = model_option


max_tokens_range = models[model_option]["tokens"]


with col2:

    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,
        max_value=max_tokens_range,
        value=4096,
        step=512,
        help=(
            "Controls the maximum length of the model's response. "
            f"Maximum for this model: {max_tokens_range}"
        )
    )


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    if message["role"] == "assistant":
        avatar = "🤖"
    else:
        avatar = "👨‍💻"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# STREAM RESPONSE GENERATOR
# --------------------------------------------------

def generate_chat_responses(
    chat_completion
) -> Generator[str, None, None]:

    """
    Streams content from the Groq API.
    """

    for chunk in chat_completion:

        content = chunk.choices[0].delta.content

        if content:
            yield content


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

if prompt := st.chat_input(
    "Enter your prompt here..."
):

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # Display user message

    with st.chat_message(
        "user",
        avatar="👨‍💻"
    ):

        st.markdown(prompt)


    # --------------------------------------------------
    # SEND REQUEST TO GROQ
    # --------------------------------------------------

    try:

        chat_completion = client.chat.completions.create(

            model=model_option,

            messages=[
                {
                    "role": message["role"],
                    "content": message["content"]
                }
                for message in st.session_state.messages
            ],

            max_tokens=max_tokens,

            stream=True
        )


        # --------------------------------------------------
        # STREAM ASSISTANT RESPONSE
        # --------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            response_generator = generate_chat_responses(
                chat_completion
            )

            full_response = st.write_stream(
                response_generator
            )


        # --------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # --------------------------------------------------

        if full_response:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": str(full_response)
                }
            )


    # --------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------

    except Exception as e:

        st.error(
            f"Groq API error: {e}",
            icon="🚨"
        )
