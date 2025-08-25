import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Sahayak — Indic Study Buddy", page_icon="📚")

# ---------- 1) SYSTEM PROMPT ----------
SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are “Sahayak”, an open-source Indic Study Buddy.
Your users are students in India who use Hindi, English, or Hinglish.
Always reply in the **same language** as the input, unless the user clearly asks for a different language.

Your tasks:
1. **Summary Mode:** Summarize text into 4–6 clear bullet points.
2. **Flashcard Mode:** Create 4–6 question-answer flashcards for revision.
3. **Quiz Mode:** Generate 4–6 multiple-choice questions (MCQs) with 4 options each and mark the correct one.

Guidelines:
- Match the input language (Hindi → Hindi, English → English, Hinglish → Hinglish).
- For Hinglish, reply in Hinglish unless the user asks for Hindi or English only.
- Keep answers concise and easy to understand.
- Use simple Hindi (Devanagari) when replying in Hindi.
- Ask a clarifying question if the input is unclear.
- Be respectful, helpful, and neutral.
- Do not give definitive medical, legal, or financial advice.
Examples:
- Input: “Explain photosynthesis.” → Output in English.
- Input: “प्रकाश संश्लेषण समझाओ।” → Output in Hindi.
- Input: “samay kya hai?” → Output in Hinglish style.
"""

def call_llm_with_fallback(system_prompt: str, history: list, user_content: str):
    """
    Tries chat API; if 404/unsupported, falls back to text_generation with a chat-like prompt.
    history: list of (user, assistant) tuples
    """
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_content})

    # 1) Try chat_completion (works if the endpoint supports it)
    try:
        out = client.chat_completion(messages=messages, max_tokens=700, temperature=0.4)
        return out.choices[0].message["content"]
    except Exception:
        pass

    # 2) Fallback: flatten to a single prompt for text_generation
    lines = []
    for m in messages:
        if m["role"] == "system":
            lines.append("System:\n" + m["content"])
        elif m["role"] == "user":
            lines.append("User:\n" + m["content"])
        elif m["role"] == "assistant":
            lines.append("Assistant:\n" + m["content"])
    prompt = "\n\n".join(lines) + "\n\nAssistant:\n"

    return client.text_generation(
        prompt,
        max_new_tokens=700,
        temperature=0.4,
        stream=False,
        return_full_text=False,
    )


# ---------- 2) MODEL CHOICE ----------
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Create a client.
# If you set a token in Streamlit secrets as HF_TOKEN, the hub lib will pick it up automatically.


MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Pick up token from secrets
HF_TOKEN = st.secrets.get("HF_TOKEN")

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)


mode = st.sidebar.radio(
    "Choose Mode",
    ["Summary", "Flashcards", "Quiz"]
)

st.title("📚 Sahayak — Indic Study Buddy")
st.caption("Type in Hindi, English, or Hinglish. Select a mode in the sidebar. This ai assistan helps students with study material.")

if "history" not in st.session_state:
    st.session_state.history = []

def call_model(user_text, history, mode):
    # System + mode instruction + history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"Mode: {mode}\n\nText: {user_text}"})

    def call_llm_with_fallback(system_prompt: str, history: list, user_content: str):
    """
    Tries chat API; if 404/unsupported, falls back to text_generation with a chat-like prompt.
    history: list of (user, assistant) tuples
    """
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_content})

    # 1) Try chat_completion (works if supported)
    try:
        out = client.chat_completion(messages=messages, max_tokens=700, temperature=0.4)
        return out.choices[0].message["content"]
    except Exception:
        pass

    # 2) Fallback: flatten messages for text_generation
    lines = []
    for m in messages:
        if m["role"] == "system":
            lines.append("System:\n" + m["content"])
        elif m["role"] == "user":
            lines.append("User:\n" + m["content"])
        elif m["role"] == "assistant":
            lines.append("Assistant:\n" + m["content"])
    prompt = "\n\n".join(lines) + "\n\nAssistant:\n"

    return client.text_generation(
        prompt,
        max_new_tokens=700,
        temperature=0.4,
        stream=False,
        return_full_text=False,
    )

    reply = call_llm_with_fallback(SYSTEM_PROMPT, st.session_state.history, f"Mode: {mode}\n\nText:\n{user_text}")
    

# ---------- INPUT ----------
with st.form("study_form", clear_on_submit=True):
    user_text = st.text_area("Paste your study material here:", height=200)
    submit = st.form_submit_button("Generate")

if submit and user_text.strip():
    with st.spinner("Sahayak is preparing your study material..."):
        reply = call_model(user_text.strip(), st.session_state.history, mode)
        st.session_state.history.append((user_text.strip(), reply))

# ---------- OUTPUT ----------
for u, a in st.session_state.history[-1:]:  # show only latest
    st.subheader(f"{mode} Output:")
    st.write(a)

st.markdown("---")
st.markdown("💡 **Tip:** Try pasting a paragraph from your textbook and switch between Summary, Flashcards, and Quiz modes.")
