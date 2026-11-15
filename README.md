# Llama 3 REST API with RAG (Django)

A Django REST Framework service that serves **Meta Llama 3 8B Instruct** behind two endpoints: a plain chat endpoint and a document question-answering endpoint that uses retrieval-augmented generation (RAG) over an uploaded PDF.

## What it does

| Endpoint | Method | Purpose |
|---|---|---|
| `/chatbot/` | POST | Send a `query`, get a Llama 3 answer. |
| `/query/` | POST | Upload a PDF `file` with a `question`. The answer is grounded in the document. |

`/query/` also accepts two optional fields: `sys_prompt` to replace the default system prompt and `botname` to set the assistant's name in the prompt.

## How the RAG endpoint works

1. The uploaded PDF is loaded with `PyPDFLoader`.
2. Text is split into 1,024-character chunks with a 64-character overlap.
3. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in an in-memory Chroma index.
4. The top 2 chunks are retrieved and passed to Llama 3 through a LangChain `RetrievalQA` chain.
5. The uploaded file is deleted after the answer is returned.

The model is loaded once at startup (`model_loader.py`) and shared across requests, so each request does not reload the weights.

## Stack

Python · Django · Django REST Framework · Hugging Face Transformers · LangChain · ChromaDB · PyTorch (CUDA)

## Run it

You need a CUDA GPU and access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face.

```bash
pip install django djangorestframework transformers torch langchain chromadb sentence-transformers pypdf
export HF_TOKEN=<your Hugging Face token>
cd Rest_api_llama3_system_prmpt/apillama3
python manage.py migrate
python manage.py runserver
```

### Example requests

```bash
curl -X POST http://127.0.0.1:8000/chatbot/ -H "Content-Type: application/json" -d '{"query": "Explain RAG in two sentences."}'
```

```bash
curl -X POST http://127.0.0.1:8000/query/ -F "file=@paper.pdf" -F "question=What problem does this paper solve?"
```

## Repository layout

```
Rest_api_llama3_system_prmpt/apillama3/
├── apillama3/          # Django project settings and root URLs
└── llama3api/
    ├── model_loader.py # loads Llama 3 once at startup
    ├── llama3file.py   # chat generation and the RAG chain
    ├── views.py        # ChatbotView and QueryView
    └── urls.py
llama_Rest_Api_ui_stremlit.zip  # small Streamlit front end for the API
```

## Author

Hamza Abbas · [LinkedIn](https://www.linkedin.com/in/hamza-abbas-851864229/) · [GitHub](https://github.com/HamzaAbbas0)
