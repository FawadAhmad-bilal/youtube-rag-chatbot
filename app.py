import os
import re
import streamlit as st

# load HF_TOKEN from a local .env file (if python-dotenv is installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


from operator import itemgetter
from urllib.parse import urlparse, parse_qs
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline,HuggingFaceEndpoint,ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, IpBlocked, RequestBlocked
from youtube_transcript_api.proxies import GenericProxyConfig
from langchain_core.runnables import RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥", layout="wide")

# ---------- dark theme css ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 900px;}
.hero {
    padding: 28px 32px; border-radius: 18px; margin-bottom: 24px;
    background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 50%, #0e7490 100%);
    border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 {margin: 0; font-size: 2rem; color: #ffffff;}
.hero p {margin: 6px 0 0 0; color: #c7d2fe; font-size: 0.95rem;}
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 0.8rem; margin-top: 10px; margin-right: 6px;
    background: rgba(255,255,255,0.12); color: #e0e7ff;
}
.empty-card {
    padding: 40px; text-align: center; border-radius: 16px;
    background: #141824; border: 1px dashed #3b3f5c; color: #9ca3c7;
}
[data-testid="stSidebar"] {border-right: 1px solid #1f2437;}
[data-testid="stChatMessage"] {
    background: #141824; border: 1px solid #1f2437;
    border-radius: 14px; padding: 12px 16px;
}
.stButton > button {
    width: 100%; border-radius: 10px; border: none; font-weight: 600;
    background: linear-gradient(90deg, #7c3aed, #0891b2); color: white;
}
.stButton > button:hover {opacity: 0.9; color: white;}
</style>
""", unsafe_allow_html=True)

# language options
LANGUAGES={
    "English":"en",
    "Hindi":"hi",
    "Urdu":"ur",
    "Arabic":"ar",
    "Spanish":"es",
    "French":"fr"
}


# accepts URL (watch, youtu.be, shorts, embed, live) or plain video ID
def get_video_id(text):
    text=text.strip()
    if "youtu.be/" in text:
        return urlparse(text).path.strip("/")
    if "youtube.com" in text:
        parsed=urlparse(text)
        if "v" in parse_qs(parsed.query):
            return parse_qs(parsed.query)["v"][0]
        parts=[p for p in parsed.path.split("/") if p]
        if len(parts)>=2 and parts[0] in ("shorts","embed","live"):
            return parts[1]
        return ""
    return text


# YouTube IDs are always 11 characters (letters, digits, - and _)
def is_valid_id(video_id):
    return re.fullmatch(r"[A-Za-z0-9_-]{11}",video_id) is not None


# multilingual embedding model (loaded once)
@st.cache_resource
def load_embadding():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


# read from Streamlit secrets, fallback to environment variable
def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


# Groq models in order of preference.
# llama-3.3-70b-versatile and llama-3.1-8b-instant were shut down on Groq (Aug 2026),
# so the app also asks Groq which models are live for your key and uses those.
GROQ_PREFERRED=[
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]
# model ids that are not chat models
GROQ_SKIP=["whisper","tts","guard","safeguard","embed","orpheus","compound"]

# HuggingFace Inference Providers models (free credits are tiny, used only as backup)
API_MODELS=[
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct"
]


# HuggingFace token: accepts HF_TOKEN or HUGGINGFACEHUB_API_TOKEN
def get_hf_token():
    return get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN")


# ask Groq which models are live for this key, pick chat models by preference
@st.cache_resource
def get_groq_models(api_key):
    try:
        from groq import Groq
        live=[m.id for m in Groq(api_key=api_key).models.list().data]
    except Exception:
        return GROQ_PREFERRED
    live=[m for m in live if not any(bad in m for bad in GROQ_SKIP)]
    ordered=[m for m in GROQ_PREFERRED if m in live]
    others=[m for m in live if m not in ordered]
    return ordered+others if (ordered+others) else GROQ_PREFERRED


# build one Groq chat model (reasoning models need extra settings)
def make_groq_chat(name,api_key):
    from langchain_groq import ChatGroq
    kwargs=dict(model=name,api_key=api_key,temperature=0.1,max_tokens=1500)
    if "gpt-oss" in name:
        kwargs["reasoning_effort"]="low"
    elif "qwen" in name:
        kwargs["reasoning_format"]="hidden"
    return ChatGroq(**kwargs)


# remove <think>...</think> blocks if a model returns them
def clean_answer(text):
    return re.sub(r"<think>.*?</think>","",text,flags=re.DOTALL).strip()


# which backend is active
def get_backend():
    if get_secret("GROQ_API_KEY"):
        return "groq"
    if get_hf_token():
        return "hf"
    return "local"


# llm (loaded once)
# GROQ_API_KEY present -> Groq (free, fast)
# HF token present     -> HuggingFace API
# nothing              -> local Qwen model (needs a lot of RAM)
@st.cache_resource
def load_model():
    backend=get_backend()

    if backend=="groq":
        key=get_secret("GROQ_API_KEY")
        names=([get_secret("GROQ_MODEL")] if get_secret("GROQ_MODEL") else [])+get_groq_models(key)[:4]
        chats=[make_groq_chat(name,key) for name in dict.fromkeys(names)]
        return chats[0].with_fallbacks(chats[1:])

    if backend=="hf":
        token=get_hf_token()
        repo_ids=([get_secret("HF_MODEL")] if get_secret("HF_MODEL") else [])+API_MODELS
        chats=[]
        for repo_id in repo_ids:
            llm=HuggingFaceEndpoint(
                repo_id=repo_id,
                task="text-generation",
                huggingfacehub_api_token=token,
                max_new_tokens=300,
                temperature=0.1
            )
            chats.append(ChatHuggingFace(llm=llm))
        return chats[0].with_fallbacks(chats[1:])

    llm=HuggingFacePipeline.from_model_id(
        model_id='Qwen/Qwen2.5-1.5B-Instruct',
        task='text-generation',
        pipeline_kwargs=dict(max_new_tokens=300,do_sample=False,return_full_text=False)
    )
    return ChatHuggingFace(llm=llm)


# indexing: transcript -> chunks -> embedding -> FAISS
@st.cache_resource
def build_vector_store(video_url,video_lang):
    # optional proxy (YouTube often blocks cloud IPs), set PROXY_URL in secrets
    proxy_url=get_secret("PROXY_URL")
    if proxy_url:
        youtueb_list=YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=proxy_url,https_url=proxy_url))
    else:
        youtueb_list=YouTubeTranscriptApi()
    transcript_list=youtueb_list.fetch(video_url,languages=list(dict.fromkeys([video_lang,'en']))).to_raw_data()

    text_flatten=" ".join(chunk['text'] for chunk in transcript_list)

    split=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n","\n","।","۔",". "," ",""]
    )
    chunk=split.create_documents([text_flatten])

    embadding=load_embadding()
    vector_store=FAISS.from_documents(chunk,embadding)
    return vector_store


def format_docs(retrieved_docs):
    context_docs="\n".join(do.page_content for do in retrieved_docs)
    return context_docs


prompt=PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer only from the provided transcript context.
        If the context is insufficient, just say you don't know
        Answer in {language}.

        {context}
        question:{question}

""",
input_variables=['context','question','language'])


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state["messages"]=[]


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    backend=get_backend()
    if backend=="groq":
        st.caption(f"🟢 Groq: {get_groq_models(get_secret('GROQ_API_KEY'))[0]}")
    elif backend=="hf":
        st.caption("🟡 Model: HuggingFace API")
    else:
        st.caption("🟠 Model: local (add GROQ_API_KEY in .env or secrets)")
    video_input=st.text_input("YouTube URL or Video ID",placeholder="https://youtube.com/watch?v=...")
    video_lang_name=st.selectbox("Video language",list(LANGUAGES.keys()))
    answer_lang_name=st.selectbox("Answer language",list(LANGUAGES.keys()))

    if st.button("Load Video"):
        video_id=get_video_id(video_input)
        if not video_id:
            st.error("Please enter a YouTube URL or video ID")
        elif not is_valid_id(video_id):
            st.error("Invalid YouTube URL or video ID")
        else:
            try:
                with st.spinner("Loading transcript..."):
                    st.session_state["vector_store"]=build_vector_store(video_id,LANGUAGES[video_lang_name])
                    st.session_state["video_id"]=video_id
                    st.session_state["messages"]=[]
                st.success("Video is ready ✅")
            except TranscriptsDisabled:
                st.error("No captions available for this video")
            except NoTranscriptFound:
                st.error("Transcript not found in this language, try another one")
            except (IpBlocked,RequestBlocked):
                st.error("YouTube blocked this server. Add PROXY_URL in secrets or run the app locally")
            except Exception as e:
                st.error(f"Error: {e}")

    if "video_id" in st.session_state:
        st.video(f"https://www.youtube.com/watch?v={st.session_state['video_id']}")
        if st.button("🗑️ Clear Chat"):
            st.session_state["messages"]=[]
            st.rerun()


# ---------- main ----------
st.markdown("""
<div class="hero">
    <h1>🎥 YouTube RAG Chatbot</h1>
    <p>Ask anything about a YouTube video. Answers come straight from its transcript.</p>
    <span class="badge">FAISS</span>
    <span class="badge">LangChain</span>
    <span class="badge">Multilingual</span>
</div>
""", unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.markdown("""
    <div class="empty-card">
        <h3>👈 Load a video first</h3>
        <p>Paste a YouTube link in the sidebar, pick a language and click <b>Load Video</b>.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    st.text(msg["sources"])

    question=st.chat_input("Ask about the video (any language)")

    if question:
        st.session_state["messages"].append({"role":"user","content":question})
        with st.chat_message("user"):
            st.markdown(question)

        # retrival process
        retriver=st.session_state["vector_store"].as_retriever(search_type='similarity',search_kwargs={'k':3})

        parallel_chain=RunnableParallel({
            'context':itemgetter('question')|retriver|RunnableLambda(format_docs),
            'question':itemgetter('question'),
            'language':itemgetter('language')
        })
        parser=StrOutputParser()
        model=load_model()

        main_chain=parallel_chain|prompt|model|parser

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    llm_ouptut=clean_answer(main_chain.invoke({'question':question,'language':answer_lang_name}))
            except Exception as e:
                st.error(f"Could not generate an answer: {e}")
                st.session_state["messages"].pop()
                st.stop()
            st.markdown(llm_ouptut)
            sources=format_docs(retriver.invoke(question))
            with st.expander("📄 Sources"):
                st.text(sources)

        st.session_state["messages"].append({"role":"assistant","content":llm_ouptut,"sources":sources})
