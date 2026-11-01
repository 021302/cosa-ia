# Reporte: Servicio de Question Answering con RoBERTa 

**Equipo:** Emma Palacios, Andre Calderon, Dennis Palacios, Rodrigo Lazo  
**Modelo:** RoBERTa-base  
**Dataset:** SQuAD v2.0  

---

## 1. Justificación del Modelo Elegido

Se seleccionó la arquitectura **RoBERTa-base** (Robustly Optimized BERT Approach) para este sistema de Question Answering debido a las siguientes razones clave:

* **Eficacia en Tareas Extractivas:** Al ser un modelo basado en el encoder de Transformer, RoBERTa genera representaciones contextuales bidireccionales de alta calidad, ideales para predecir las posiciones de inicio (`start`) y fin (`end`) de las respuestas dentro del texto suministrado.
* **Optimización en SQuAD v2.0:** A diferencia de BERT tradicional, RoBERTa omitió el objetivo de predicción de la siguiente oración (NSP) y fue entrenado con secuencias más largas y batches más grandes, logrando un rendimiento superior en la detección de preguntas sin respuesta (*unanswerable questions*).
* **Eficiencia de Inferencia (CPU):** Cuenta con 125 millones de parámetros, lo que permite un balance idóneo entre alta precisión semántica y baja latencia de respuesta en entornos de ejecución con recursos limitados (CPU de laptop).

---

## 2. Proceso de Entrenamiento e Hiperparámetros

El modelo fue ajustado de forma fina (*fine-tuning*) sobre el dataset SQuAD v2.0 utilizando la librería `transformers` de Hugging Face.

### Hiperparámetros de Entrenamiento:
* **Learning Rate:** `3e-5` (con decaimiento lineal)
* **Batch Size:** `16`
* **Épocas:** `3`
* **Max Sequence Length:** `384`
* **Doc Stride:** `128`
* **Optimizer:** AdamW (`weight_decay=0.01`)

### Curvas y Dinámica de Entrenamiento:
Durante la primera época, la pérdida de entrenamiento (*training loss*) disminuyó drásticamente de ~2.4 a ~1.1, alcanzando una convergencia estable en la tercera época con una pérdida de validación de ~0.85. El manejo de contextos largos mediante la estrategia de ventana deslizante (*sliding window* con `doc_stride=128`) garantizó que no se perdiera contexto relevante en párrafos extensos.

---

## 3. Umbral para Preguntas Sin Respuesta (*Unanswerable Questions*)

SQuAD v2.0 incluye preguntas que no pueden ser respondidas con el texto provisto. Para lidiar con esto:

1. **Token nulo (`[CLS]`):** La probabilidad de que una pregunta no tenga respuesta se calcula mediante el producto de las probabilidades softmax del primer token (`index 0`).
2. **Estrategia de Umbral:** Se definió un criterio donde, si la probabilidad conjunta del token nulo ($S_{null}$) supera la mejor probabilidad de respuesta extraíble ($S_{best}$) y además cumple con $S_{null} > 0.40$, la API retorna `answer: null`, `start: null`.
3. **Calibración:** Este umbral de `0.40` redujo significativamente los falsos positivos en el conjunto de validación, evitando que el modelo forzara respuestas incorrectas cuando el texto no contenía la información solicitada.

---

## 4. Arquitectura del Servicio y Despliegue

La API de producción fue desarrollada con **FastAPI** y estructurada bajo las siguientes especificaciones técnicas:

* **Seguridad y Autenticación:** Se implementó middleware de validación con encabezado `Authorization: Bearer <TOKEN>`, consumiendo de forma dinámica la variable de entorno `QA_API_TOKEN`. En caso de token inválido o ausente, la API responde con la estructura estándar de error `{"error": "unauthorized", ...}`.
* **Interfaz de Prueba Integrada:** Incluye un dashboard visual minimalista en la ruta raíz (`/`) renderizado vía HTML/JS en modo oscuro, facilitando pruebas visuales de latencia, resaltado de texto en tiempo real y puntuación de confianza.
* **Tunneling Público:** Se configuró **Cloudflare Tunnel** (`cloudflared`) para exponer el puerto local `8000` a una URL pública con HTTPS seguro, permitiendo el consumo externo y la evaluación automatizada por parte de clientes remotos.

---

## 5. Investigación: Decodificadores y Generación de Respuestas

### ¿Cómo abordaría este problema un modelo basado en solo decodificadores (ej. GPT) o Encoder-Decoder (ej. T5/BART)?
A diferencia de RoBERTa (que es puramente extractivo), las arquitecturas auto-regresivas o Sequence-to-Sequence generan texto token por token de manera generativa:
* **Ventajas:** Pueden sintetizar respuestas, parafrasear y redactar explicaciones fluidas en lugar de limitarse a copiar un fragmento exacto del texto.
* **Desventajas en QA Estricto:** Tienen riesgo de alucinación (inventar datos) y requieren mayor poder de cómputo para inferencia.

### Estrategias para Textos Largos que Superan el Límite de Tokens:
1. **RAG (Retrieval-Augmented Generation):** Dividir el documento en fragmentos (*chunks*), generar embeddings con un modelo de recuperación (ej. `text-embedding-ada`) y pasar solo los fragmentos más relevantes al prompt del decodificador.
2. **Ventanas de Atención Extendida (Longform Transformers):** Usar arquitecturas como Longformer o Sparse Attention para procesar secuencias de más de 4096 tokens sin el costo cuadrático de la atención estándar.
3. **Map-Reduce Strategy:** Pasar fragmentos del documento en paralelo al decodificador para extraer respuestas parciales y luego combinarlas en una llamada final de síntesis.