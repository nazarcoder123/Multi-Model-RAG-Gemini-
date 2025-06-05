import os
import io
import tempfile
from PIL import Image
import pdf2image
import PyPDF2
import pickle
import numpy as np
import faiss
# from langchain.vectorstores import FAISS
import logging

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def pdf_to_images(pdf_file):
    logging.info("Converting PDF to images.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name

    images = pdf2image.convert_from_path(tmp_path, dpi=200)
    os.unlink(tmp_path)
    logging.info(f"Converted PDF to {len(images)} images.")
    return images

def extract_text_from_pdf(pdf_file):
    logging.info("Extracting text from PDF.")
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        logging.info("Text extraction successful.")
        return text
    except Exception as e:
        logging.error(f"Text extraction error: {e}")
        print(f"Text extraction error: {e}")
        return ""

def save_image_preview(image, filename):
    logging.info(f"Saving image preview: {filename}")
    path = os.path.join(DATA_DIR, filename)
    image.save(path)
    logging.info(f"Image preview saved to: {path}")
    return path

def save_embeddings_and_info(embeddings_data, docs_info):
    logging.info("Saving embeddings and document info.")
    if not embeddings_data:
        logging.warning("No embeddings data to save.")
        # No embeddings to save, just save docs_info
        with open(os.path.join(DATA_DIR, "docs_info.pkl"), "wb") as f:
            pickle.dump(docs_info, f)
        logging.info("Saved docs_info only.")
        return

    vectors = [item["embedding"].astype("float32") for item in embeddings_data]
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.vstack(vectors))
    faiss.write_index(index, os.path.join(DATA_DIR, "faiss.index"))
    logging.info("Saved FAISS index.")

    with open(os.path.join(DATA_DIR, "docs_info.pkl"), "wb") as f:
        pickle.dump(docs_info, f)
    logging.info("Saved docs_info.")

def load_embeddings_and_info():
    logging.info("Loading embeddings and document info.")
    index_path = os.path.join(DATA_DIR, "faiss.index")
    docs_path = os.path.join(DATA_DIR, "docs_info.pkl")

    if os.path.exists(index_path) and os.path.exists(docs_path):
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs_info = pickle.load(f)
        logging.info("Loaded existing FAISS index and docs_info.")
    else:
        index = None
        docs_info = []
        logging.warning("No existing FAISS index or docs_info found. Starting fresh.")

    return index, docs_info