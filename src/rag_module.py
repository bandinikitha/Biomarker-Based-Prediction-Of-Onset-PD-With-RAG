# src/rag_module.py obtained from gemini
import os
import json
import requests
from typing import Dict, List, Any
import streamlit as st

# --- LangChain/Chroma Imports for Alignment ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
# Config (Aligning with Ingestion Script)
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
EMBED_MODEL = "all-MiniLM-L6-v2"  # Must match ingestion script
GROQ_API_KEY = os.getenv(
    # Ensure this is correct
    "GROQ_API_KEY", "")
# Changed to use a newer model
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# --- Cached Resources ---
_embedder = None
_vector_store = None


def load_vector_store():
    """Load the Chroma DB created by the ingestion script."""
    global _embedder, _vector_store
    if _vector_store is None:
        if _embedder is None:
            # Use HuggingFaceEmbeddings as used in the ingestion script
            _embedder = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={'device': 'cpu'}
            )

        # Load the existing Chroma collection
        _vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=_embedder,
        )
    return _vector_store


# Utility: LLM call (Groq)
@st.cache_data(show_spinner="Calling Groq API to generate report...")
def call_groq(system_prompt: str, user_prompt: str, model: str = DEFAULT_GROQ_MODEL,
              max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
    """Calls the Groq API and returns the full JSON response."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        raise RuntimeError(
            "❌ GROQ_API_KEY not found in environment variables.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Main helper: retrieve top-k using LangChain's retriever interface


def retrieve_top_k(query: str, top_k: int = 4) -> List[Document]:
    """Retrieves context from the Chroma vector store."""
    if not query or not query.strip():
        return []

    # Get the LangChain-wrapped vector store and use its retriever
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Use the retriever to fetch relevant documents
    docs = retriever.invoke(query)
    return docs

# small helper to convert structured features to text


def structured_to_text(features: Dict) -> str:
    lines = [f"{k}: {v}" for k, v in features.items()]
    return "\n".join(lines)


# Prompts (Enhanced for detailed reports without renaming variables)

SYSTEM_PROMPT_PATIENT = (
    "Start the report with Dear Patient and do not use any names"
    "You are a compassionate and knowledgeable medical assistant. Produce a **detailed (around 300–400 words)**, "
    "clear, and empathetic patient-facing report of the findings and suggestions. "
    "Explain what the prediction means in simple terms, and how it relates to the patient's health. "
    "If the person has a high probability of having Parkinson's disease progression, then tell them that they have a risk of getting the disease in future  "
    "If the person has low probability of having Parkinson's disease progression tell them that they have low risk of getting the disease"
    "Use an encouraging and educational tone throughout. "
    "Avoid dense medical jargon, but do not oversimplify important details. "
    "Provide **3–5 actionable lifestyle or monitoring recommendations**, and conclude with a calm and supportive closing message. "
    "Ensure the explanation feels personalized and trustworthy."

)

SYSTEM_PROMPT_DOCTOR = (
    "You are an advanced clinical decision support assistant. Produce a **detailed, structured, and technical report** "
    "intended for a medical professional. Your report must include: "
    "1) A comprehensive summary of patient features and relevant biomarkers, and determine the impact and effect of the feature in detail "
    "2) A clear statement of the model prediction, probability, and clinical significance in detail, "
    "3) An interpretation of the top contributing features with reasoning on their physiological or clinical impact in detail, "
    "4) Evidence-based recommendations for further evaluation (tests, referrals, or monitoring), "
    "5) A brief discussion on the reliability or limitations of the prediction, and "
    "6) Proper citation of the retrieved document IDs used for reasoning. "
    "Maintain a formal, academic tone suitable for clinical documentation, and provide in-depth yet concise analysis."
)

PATIENT_USER_TEMPLATE = """Patient summary:
{patient_summary}

Model prediction: {prediction} (probability: {probability})

Relevant retrieved documents (excerpts & sources):
{retrieved_docs}

Please produce a **detailed, structured, and empathetic** patient-facing report as described in the system prompt. 
Use clear headings (e.g., 'Understanding Your Results', 'Next Steps', 'Our Recommendations'), and ensure it is written 
in reassuring, accessible language.
"""

DOCTOR_USER_TEMPLATE = """Patient summary:
{patient_summary}

Model prediction: {prediction} (probability: {probability})

Top feature importances:
{feature_importance}

Retrieved documents (excerpts & sources):
{retrieved_docs}

Please produce a **detailed, evidence-based clinician report** as described in the system prompt. 
Include subsections like 'Summary of Findings', 'Model Interpretation', 'Key Biomarkers', and 'Clinical Recommendations'. 
Maintain professional tone and clarity throughout.
"""


# Main helper: generate both reports
def generate_rag_reports(features: Dict, prediction: Dict, clinician_notes: str = "", top_k: int = 4):
    """Generates two reports using RAG and the Groq API."""
    patient_summary = structured_to_text(features)
    query = patient_summary + \
        ("\nNotes: " + clinician_notes if clinician_notes else "")

    # Use the new LangChain-aligned retrieve_top_k function
    retrieved_docs_lc = retrieve_top_k(query, top_k=top_k)

    # Format retrieved documents for the LLM prompt
    retrieved_texts = []
    for d in retrieved_docs_lc:
        excerpt = d.page_content[:400].replace("\n", " ")
        # LangChain stores metadata
        src = d.metadata.get("source", "-").split(os.path.sep)[-1]
        retrieved_texts.append(
            f"Source:{src} | Page:{d.metadata.get('page', '-')} | excerpt: {excerpt}...")

    retrieved_joined = "\n\n".join(
        retrieved_texts) if retrieved_texts else "No supporting documents found."

    # --- Prepare User Prompts ---
    patient_user = PATIENT_USER_TEMPLATE.format(
        patient_summary=patient_summary,
        prediction=prediction.get("label", "N/A"),
        probability=f"{prediction.get('probability', 'N/A'):.4f}",
        retrieved_docs=retrieved_joined
    )
    doctor_user = DOCTOR_USER_TEMPLATE.format(
        patient_summary=patient_summary,
        prediction=prediction.get("label", "N/A"),
        probability=f"{prediction.get('probability', 'N/A'):.4f}",
        feature_importance=json.dumps(
            prediction.get("importance", {}), indent=2),
        retrieved_docs=retrieved_joined
    )

    # --- Call Groq (and extract the generated text) ---
    patient_response = call_groq(
        SYSTEM_PROMPT_PATIENT, patient_user) if GROQ_API_KEY else {"choices": [{"message": {"content": "[No GROQ_KEY]"}}]}
    doctor_response = call_groq(
        SYSTEM_PROMPT_DOCTOR, doctor_user) if GROQ_API_KEY else {"choices": [{"message": {"content": "[No GROQ_KEY]"}}]}

    # Extract the generated text content
    patient_report_text = patient_response["choices"][0]["message"]["content"]
    doctor_report_text = doctor_response["choices"][0]["message"]["content"]

    return {
        "patient_report": patient_report_text,
        "doctor_report": doctor_report_text,
        # Return metadata for context
        "retrieved": [doc.metadata for doc in retrieved_docs_lc]
    }
