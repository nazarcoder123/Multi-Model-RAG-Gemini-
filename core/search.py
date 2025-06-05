import numpy as np
from PIL import Image
from config import GEMINI_API_KEY, GEMINI_MODEL
import google.generativeai as genai
import logging

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_client = genai

def search_documents(query, index, docs_info, query_embed_fn, top_k=3):
    logging.info(f"Searching documents for query: {query[:50]}...")
    query_vector = query_embed_fn(query)
    if query_vector is None or index is None:
        logging.warning("Query vector is None or index is None. Returning empty results.")
        return []

    D, I = index.search(np.array([query_vector.astype("float32")]), top_k)
    logging.info(f"FAISS search completed. Found {len(I[0])} potential matches.")
    results = []

    for score, idx in zip(D[0], I[0]):
        if idx < len(docs_info):
            doc_info = docs_info[idx]
            results.append({
                "doc_id": doc_info["doc_id"],
                "source": doc_info["source"],
                "content_type": doc_info["content_type"],
                "page": doc_info.get("page", 1),
                "similarity": 1 / (1 + score),
                "content": doc_info.get("content"),
                "preview": doc_info.get("preview"),
            })
            logging.debug(f"Added result: doc_id={doc_info['doc_id']}, similarity={1 / (1 + score)}")

    logging.info(f"Returning {len(results)} search results.")
    return results

def answer_with_gemini(question, content):
    logging.info(f"Generating answer with Gemini for question: {question[:50]}...")
    try:
        model = gemini_client.GenerativeModel(GEMINI_MODEL)

        if isinstance(content, Image.Image):
            logging.info("Answering with Gemini using image content.")
            prompt = [f"""Answer the question based on the following image.
Don't use markdown.
Please provide enough context for your answer.

Question: {question}""", content]
            response = model.generate_content(contents=prompt)
        else:
            logging.info("Answering with Gemini using text content.")
            prompt = f"""Answer the question based on the following information.
Don't use markdown.
Please provide enough context for your answer.

Information: {content}

Question: {question}"""
            response = model.generate_content(prompt)

        answer = response.text
        logging.info("Gemini answer generated.")
        print("LLM Answer:", answer)
        return answer.strip() if answer else "Gemini returned no answer."
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        print("Gemini error:", str(e))
        return f"Gemini error: {e}"