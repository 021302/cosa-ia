import os
import time
import threading
import webbrowser
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 1. CONFIGURACIÓN Y SEGURIDAD (Leyendo token del entorno)
API_TOKEN = os.environ.get("QA_API_TOKEN", "quierovacaciones")
MODEL_PATH = "./fine_tuned_qa_model"

app = FastAPI(title="QA API - SQuAD 2.0")

# Manejador de errores estricto según contrato del PDF
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    code_map = {401: "unauthorized", 400: "bad_request", 422: "bad_request", 500: "internal_error"}
    return JSONResponse(status_code=exc.status_code, content={"error": code_map.get(exc.status_code, "internal_error"), "detail": str(exc.detail)})

def abrir_navegador():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=abrir_navegador).start()

print("Cargando modelo RoBERTa entrenado...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
model.eval()

# Validador de Token Bearer obligatorio para /info y /predict
def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token no proporcionado o formato inválido")
    token = authorization.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token no válido")
    return token

# Esquemas de datos con el campo 'id' obligatorio
class QAInput(BaseModel):
    id: str
    context: str
    question: str

class QAResponse(BaseModel):
    id: str
    answer: Optional[str]
    start: Optional[int]
    score: float


# 2. INTERFAZ GRÁFICA (Tu diseño minimalista y elegante en Modo Oscuro)
@app.get("/", response_class=HTMLResponse)
def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SQuAD 2.0 | Neural QA Engine</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #090d16;
                --surface: #111827;
                --surface-border: #1f293d;
                --text-primary: #f3f4f6;
                --text-secondary: #9ca3af;
                --accent: #6366f1;
                --accent-hover: #4f46e5;
            }
            body {
                background-color: var(--bg-main);
                color: var(--text-primary);
                font-family: 'Plus Jakarta Sans', sans-serif;
                min-height: 100vh;
            }
            .dashboard-header { padding: 2.5rem 0 1.5rem 0; }
            .badge-tag {
                background: rgba(99, 102, 241, 0.1);
                color: #818cf8;
                border: 1px solid rgba(99, 102, 241, 0.2);
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                padding: 6px 14px;
                border-radius: 30px;
                font-weight: 600;
            }
            .card-panel {
                background-color: var(--surface);
                border: 1px solid var(--surface-border);
                border-radius: 20px;
                padding: 2rem;
                box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            }
            .form-label {
                color: var(--text-secondary);
                font-size: 0.85rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }
            .form-control {
                background-color: #0d1322;
                border: 1px solid var(--surface-border);
                color: var(--text-primary);
                border-radius: 12px;
                padding: 0.85rem;
                font-size: 0.95rem;
            }
            .form-control:focus {
                background-color: #0d1322;
                border-color: var(--accent);
                color: var(--text-primary);
                box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
            }
            .btn-action {
                background-color: var(--accent);
                border: none;
                border-radius: 12px;
                padding: 0.85rem;
                font-weight: 600;
                color: white;
                width: 100%;
                transition: background-color 0.2s ease;
            }
            .btn-action:hover { background-color: var(--accent-hover); color: white; }
            .metric-card {
                background-color: #0d1322;
                border: 1px solid var(--surface-border);
                border-radius: 14px;
                padding: 1rem;
                text-align: center;
            }
            .metric-label { font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 0.25rem; }
            .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600; }
            .answer-box-success {
                background: rgba(16, 185, 129, 0.05);
                border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 14px;
                padding: 1.25rem;
            }
            .answer-box-null {
                background: rgba(239, 68, 68, 0.05);
                border: 1px solid rgba(239, 68, 68, 0.2);
                border-radius: 14px;
                padding: 1.25rem;
            }
            .context-viewer {
                background-color: #0d1322;
                border: 1px solid var(--surface-border);
                border-radius: 14px;
                padding: 1.25rem;
                color: var(--text-secondary);
                font-size: 0.9rem;
                line-height: 1.6;
                max-height: 200px;
                overflow-y: auto;
            }
            .highlight { background-color: #fbbf24; color: #000; font-weight: 600; padding: 2px 4px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container py-4" style="max-width: 1100px;">
            <header class="dashboard-header text-center">
                <span class="badge-tag mb-2 d-inline-block">RoBERTa-base • SQuAD v2.0 API</span>
                <h1 class="fw-bold fs-3 mb-1">Responde preguntas :) </h1>
                <p class="text-secondary small"></p>
            </header>

            <div class="row g-4">
                <div class="col-lg-6">
                    <div class="card-panel h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h2 class="h6 fw-semibold mb-3 text-light">Parámetros de Solicitud</h2>
                            <div class="mb-3">
                                <label class="form-label">ID de Consulta</label>
                                <input type="text" id="query-id" class="form-control" value="q-017">
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Token (Bearer)</label>
                                <input type="password" id="token" class="form-control" value="quierovacaciones">
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Contexto</label>
                                <textarea id="context" class="form-control" rows="5" placeholder="Pega el texto aquí..."></textarea>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Pregunta</label>
                                <input type="text" id="question" class="form-control" placeholder="¿Qué deseas consultar?">
                            </div>
                        </div>
                        <button onclick="consultarAPI()" class="btn-action">Ejecutar Predicción</button>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="card-panel h-100 d-flex flex-column justify-content-between">
                        <div>
                            <h2 class="h6 fw-semibold mb-3 text-light">Resultado de Inferencia</h2>
                            <div id="loading" class="text-center py-5 d-none">
                                <div class="spinner-border text-indigo" style="color: #6366f1;" role="status"></div>
                                <p class="mt-2 text-secondary small">Procesando modelo...</p>
                            </div>
                            <div id="result-box" class="d-none">
                                <div id="answer-card" class="mb-3"></div>
                                <div class="row g-2 mb-3">
                                    <div class="col-4">
                                        <div class="metric-card">
                                            <div class="metric-label">Confianza</div>
                                            <div id="score-badge" class="metric-value text-info" style="color: #818cf8;">-</div>
                                        </div>
                                    </div>
                                    <div class="col-4">
                                        <div class="metric-card">
                                            <div class="metric-label">Posición (Start)</div>
                                            <div id="start-badge" class="metric-value">-</div>
                                        </div>
                                    </div>
                                    <div class="col-4">
                                        <div class="metric-card">
                                            <div class="metric-label">Latencia</div>
                                            <div id="latency-badge" class="metric-value text-success" style="color: #34d399;">-</div>
                                        </div>
                                    </div>
                                </div>
                                <label class="form-label mb-2">Contexto Resaltado</label>
                                <div id="highlighted-context" class="context-viewer"></div>
                            </div>
                            <div id="placeholder-box" class="text-center py-5 text-secondary my-auto">
                                <p class="small">Ingresa una consulta a la izquierda para ver la respuesta del modelo.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function consultarAPI() {
                const id = document.getElementById('query-id').value.trim();
                const token = document.getElementById('token').value.trim();
                const context = document.getElementById('context').value.trim();
                const question = document.getElementById('question').value.trim();

                if (!context || !question) {
                    alert("Complete el contexto y la pregunta.");
                    return;
                }

                document.getElementById('loading').classList.remove('d-none');
                document.getElementById('result-box').classList.add('d-none');
                document.getElementById('placeholder-box').classList.add('d-none');

                const startTime = performance.now();
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + token
                        },
                        body: JSON.stringify({ id, context, question })
                    });
                    const endTime = performance.now();
                    const latency = Math.round(endTime - startTime);

                    if (!response.ok) {
                        const err = await response.json();
                        alert("Error [" + response.status + "]: " + (err.error || "Error"));
                        document.getElementById('loading').classList.add('d-none');
                        document.getElementById('placeholder-box').classList.remove('d-none');
                        return;
                    }

                    const data = await response.json();
                    document.getElementById('loading').classList.add('d-none');
                    document.getElementById('result-box').classList.remove('d-none');

                    document.getElementById('latency-badge').innerText = latency + " ms";
                    document.getElementById('score-badge').innerText = (data.score * 100).toFixed(1) + "%";

                    const answerCard = document.getElementById('answer-card');
                    const startBadge = document.getElementById('start-badge');
                    const highlightedContext = document.getElementById('highlighted-context');

                    if (data.answer === null) {
                        answerCard.className = "answer-box-null";
                        answerCard.innerHTML = "<span class='text-danger fw-semibold small'>Sin Respuesta (Unanswerable)</span>";
                        startBadge.innerText = "N/A";
                        highlightedContext.innerText = context;
                    } else {
                        answerCard.className = "answer-box-success";
                        answerCard.innerHTML = "<span class='text-success fw-semibold small'>Respuesta Extraída (ID: " + data.id + ")</span><div class='fw-bold text-light mt-1'>" + data.answer + "</div>";
                        startBadge.innerText = data.start;

                        const before = context.substring(0, data.start);
                        const match = context.substring(data.start, data.start + data.answer.length);
                        const after = context.substring(data.start + data.answer.length);
                        highlightedContext.innerHTML = before + "<mark class='highlight'>" + match + "</mark>" + after;
                    }
                } catch (e) {
                    alert("Error de conexión con la API.");
                    document.getElementById('loading').classList.add('d-none');
                    document.getElementById('placeholder-box').classList.remove('d-none');
                }
            }
        </script>
    </body>
    </html>
    """


# 3. ENDPOINTS DE LA API (Cumpliendo estrictamente el contrato del PDF)

@app.get("/health")
def health_check():
    """Público, responde 200 OK si el modelo cargó[cite: 4, 10]."""
    return {"status": "ok"}

@app.get("/info")
def model_info(token: str = Depends(verify_token)):
    """Protegido con token obligatorio[cite: 4, 10]."""
    num_params = sum(p.numel() for p in model.parameters())
    return {
        "team": "team-03",
        "model": "roberta-base",
        "params": num_params,
        "max_seq_length": 384,
        "doc_stride": 128
    }

@app.post("/predict", response_model=QAResponse)
def predict(data: QAInput, token: str = Depends(verify_token)):
    """Protegido y recibe/devuelve el campo id exigido[cite: 5, 10]."""
    context = data.context
    question = data.question
    query_id = data.id

    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        stride=128,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    offset_mapping = inputs.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)

    score_null = (start_probs[0] * end_probs[0]).item()

    best_score = 0.0
    best_start = 0
    best_end = 0

    sequence_ids = inputs.sequence_ids(0)
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while idx < len(sequence_ids) and sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    for start_i in range(context_start, context_end + 1):
        for end_i in range(start_i, min(start_i + 30, context_end + 1)):
            score = (start_probs[start_i] * end_probs[end_i]).item()
            if score > best_score:
                best_score = score
                best_start = start_i
                best_end = end_i

    # Umbral flexible optimizado para español
    if score_null > best_score and score_null > 0.4:
        return QAResponse(id=query_id, answer=None, start=None, score=round(score_null, 2))

    start_char = int(offset_mapping[best_start][0])
    end_char = int(offset_mapping[best_end][1])
    extracted_answer = context[start_char:end_char]

    return QAResponse(
        id=query_id,
        answer=extracted_answer,
        start=start_char,
        score=round(best_score, 2)
    )