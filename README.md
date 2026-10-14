# 🎥 YouTube RAG Chatbot

Ask questions about any YouTube video. The app pulls the video transcript, finds the most relevant parts, and answers only from them.

Works with English, Hindi, Urdu, Arabic, Spanish and French. You can ask in one language and get the answer in another.

## How it works

```
YouTube transcript → split into chunks → multilingual embeddings → FAISS
Question → retrieve top 3 chunks → prompt → LLM → answer (+ sources)
```

- **Chunking:** `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200, Hindi `।` and Urdu `۔` separators)
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (runs locally)
- **Vector store:** FAISS
- **LLM:** Groq API (default). Falls back to the HuggingFace API, then to a local Qwen model if no key is set.

## Tech stack

Python, Streamlit, LangChain, FAISS, Sentence Transformers, Groq, `youtube-transcript-api`

## Run locally

```bash
git clone https://github.com/FawadAhmad-bilal/youtube-rag-chatbot.git
cd youtube-rag-chatbot

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

Create a `.env` file next to `app.py`:

```
GROQ_API_KEY=your_groq_key
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys), then run:

```bash
streamlit run app.py
```

GROQ_API_KEY = "your_groq_key"
```

See `.streamlit/secrets.toml.example` for all optional settings.

## Notes

- YouTube sometimes blocks cloud server IPs. If transcripts fail after deploying, add a `PROXY_URL` secret or run the app locally.
- Groq free tier has rate limits. The app automatically tries the next available model if one fails.
- Without any API key the app uses a local Qwen model, which needs a lot of RAM and is slow on CPU.

## Project structure

```
├── app.py
├── requirements.txt
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```
