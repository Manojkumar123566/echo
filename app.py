import os
import base64
import requests
import streamlit as st
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# =============== CONFIG ===============
# Read credentials from environment variables or user input
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", st.sidebar.text_input("Hugging Face API Token", type="password"))
TTS_API_KEY = os.getenv("IBM_TTS_API_KEY", st.sidebar.text_input("IBM TTS API Key", type="password"))
TTS_URL = os.getenv("IBM_TTS_URL", st.sidebar.text_input("IBM TTS URL"))

# Hugging Face Granite instruct model (hosted by IBM on HF)
GRANITE_MODEL = "ibm-granite/granite-8b-instruct"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{GRANITE_MODEL}"

# =============== SESSION STATE ===============
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# =============== STREAMLIT PAGE SETUP ===============
st.set_page_config(page_title="🎤 EchoVerse", layout="wide")

st.markdown("""
<style>
/* Background */
.stApp {
  background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
  color: #333;
  font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
}

/* Headings */
h1, h2, h3 {
  text-align: center;
  color: #fff;
  font-weight: 800;
}

/* Section cards */
.section {
  background: #ffffff;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  margin-bottom: 20px;
}

/* Textareas */
textarea {
  background: #f3f4f6 !important;
  border-radius: 8px !important;
  font-size: 1.1em;
}

/* File Uploader */
.stFileUploader label {
  color: #4CAF50;
}

/* Buttons */
.stButton > button {
  background: #5C6BC0;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 10px 16px;
  font-weight: 600;
  box-shadow: 0 4px 10px rgba(0,0,0,0.1);
  transition: transform .1s ease-in-out;
}
.stButton > button:hover {
  transform: translateY(-2px);
  background: #3F51B5;
}

/* Clear Button */
.clear-button > button {
  background: #EF5350;
  color: #fff;
}

/* Text boxes */
.text-box {
  background: #f9fafb;
  border-radius: 12px;
  padding: 16px;
  color: #333;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  font-size: 1.2em;
}
.original { border-left: 5px solid #FFB74D; }
.adapted  { border-left: 5px solid #66BB6A; }

/* Audio */
audio {
  width: 100%;
  border-radius: 8px;
  margin-top: 10px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Radio buttons */
.stRadio > label {
  background: rgba(255,255,255,0.9);
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 10px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("🎤 EchoVerse")
st.subheader("Transform your text into expressive audiobooks with AI-powered tone adaptation")

# =============== UI ===============
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Input Your Text")
    uploaded_file = st.file_uploader("Upload Text File (Max 20MB)", type=["txt"])
    if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
        try:
            st.session_state.text = uploaded_file.read().decode('utf-8')
            st.session_state.file_name = uploaded_file.name
        except:
            st.warning("Could not read the file. Please ensure it's a valid text file.")
    st.markdown("Or drag and drop your .txt file here")
    st.subheader("Or Paste Your Text Here")
    st.session_state.text = st.text_area("", value=st.session_state.text, height=200, placeholder="Enter your text here...")
    col_char, col_clear = st.columns([4, 1])
    with col_char:
        st.markdown(f"**Characters:** {len(st.session_state.text)}")
    with col_clear:
        if st.button("Clear", key="clear"):
            st.session_state.text = ""
            st.session_state.file_name = None
    st.markdown('</div>', unsafe_allow_html=True)

col_tone, col_voice = st.columns(2)

with col_tone:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Choose Tone")
    tone_options = [
        "Neutral - Professional and clear narration",
        "Suspenseful - Dramatic and engaging delivery",
        "Inspiring - Motivational and uplifting tone"
    ]
    tone_choice = st.radio("", tone_options)
    tone = tone_choice.split(" - ")[0]
    st.markdown('</div>', unsafe_allow_html=True)

with col_voice:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Select Voice")
    voice_options = [
        "Lisa - Warm female voice",
        "Michael - Authoritative male voice",
        "Allison - Friendly female voice"
    ]
    voice_label = st.radio("", voice_options)
    VOICE_MAP = {
        "Lisa": "en-US_LisaV3Voice",
        "Michael": "en-US_MichaelV3Voice",
        "Allison": "en-US_AllisonV3Voice",
    }
    voice = VOICE_MAP[voice_label.split(" - ")[0]]
    st.markdown('</div>', unsafe_allow_html=True)

# =============== HELPERS ===============
def rewrite_with_granite_hf(text: str, tone_choice: str) -> str:
    if not HF_TOKEN:
        st.warning("HUGGINGFACE_API_TOKEN is not set. Returning original text.")
        return text

    system = (
        "You are an expert editor. Rewrite user text in the requested tone while preserving factual meaning, "
        "keeping names/dates intact, improving clarity and flow, and enhancing expressiveness. "
        "Do not add new facts. Keep the output roughly the same length as the input."
    )
    user = f"Tone: {tone_choice}\n\nText:\n{text}"

    prompt = f"""<|system|>
{system}
<|user|>
{user}
<|assistant|>
"""

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "return_full_text": False
        }
    }

    try:
        resp = requests.post(HF_INFERENCE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return text
    except Exception as e:
        st.error(f"Granite rewrite failed: {e}")
        return text

def synthesize_with_watson_tts(text: str, voice_id: str) -> bytes:
    if not (TTS_API_KEY and TTS_URL):
        st.error("IBM_TTS_API_KEY or IBM_TTS_URL not set. Please configure environment variables or enter them in the sidebar.")
        return None

    authenticator = IAMAuthenticator(TTS_API_KEY)
    tts = TextToSpeechV1(authenticator=authenticator)
    tts.set_service_url(TTS_URL)

    try:
        audio_content = tts.synthesize(
            text=text,
            voice=voice_id,
            accept="audio/mp3"
        ).get_result().content
        return audio_content
    except Exception as e:
        st.error(f"TTS synthesis failed: {e}")
        return None

# =============== ACTION ===============
go = st.button("Transform Text to Audio")

if go:
    user_text = st.session_state.text
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("✨ Rewriting with IBM Granite (Hugging Face)…"):
            adapted_text = rewrite_with_granite_hf(user_text, tone)

        audio_bytes = None
        try:
            with st.spinner(f"🔊 Synthesizing voice with IBM Watson TTS ({voice_label.split(' - ')[0]})…"):
                audio_bytes = synthesize_with_watson_tts(adapted_text, voice)
        except Exception as e:
            st.error(f"TTS failed: {e}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="text-box original"><h3>📝 Original Text</h3>', unsafe_allow_html=True)
            st.write(user_text)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="text-box adapted"><h3>🎭 Adapted Text ({tone})</h3>', unsafe_allow_html=True)
            st.write(adapted_text)
            st.markdown('</div>', unsafe_allow_html=True)

        if audio_bytes:
            st.subheader("🔊 Your Audiobook")
            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            st.markdown(
                f"""
                <audio controls>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """,
                unsafe_allow_html=True
            )
            st.download_button(
                "⬇ Download MP3",
                data=audio_bytes,
                file_name="audiobook.mp3",
                mime="audio/mp3"
            )
            st.success("✅ Audiobook generated successfully!")
        else:
            st.error("Failed to generate audio. Check environment variables and API credentials.")

# =============== NOTES ===============
st.markdown(
    """
    <br>
    <div class="section" style="color:#333;">
      <b>Notes</b><br>
      • IBM Granite is an LLM for text generation; it does not provide TTS. We use IBM <i>Watson Text-to-Speech</i> for natural voices (Lisa, Michael, Allison).<br>
      • The Granite model is accessed via the Hugging Face Inference API. Set <code>HUGGINGFACE_API_TOKEN</code> in your environment or sidebar.<br>
      • Set <code>IBM_TTS_API_KEY</code> and <code>IBM_TTS_URL</code> for Watson TTS in your environment or sidebar.<br>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<p style='text-align: center; color: #fff;'>Made with Genspark</p>", unsafe_allow_html=True)