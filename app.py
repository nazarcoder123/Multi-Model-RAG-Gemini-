import streamlit as st
from PIL import Image
import uuid
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

from core.embeddings import get_document_embedding, get_query_embedding
from core.document_utils import (
    pdf_to_images,
    extract_text_from_pdf,
    save_image_preview,
    load_embeddings_and_info,
    save_embeddings_and_info,
)
from core.search import search_documents, answer_with_gemini

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load session state
logging.info("Initializing session state")
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index, st.session_state.docs_info = load_embeddings_and_info()
    logging.info("Loaded embeddings and docs_info from file.")
if 'embedding_buffer' not in st.session_state:
    st.session_state.embedding_buffer = []

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal Search App 🔍")

tab1, tab2 = st.tabs(["Index Documents", "Search"])

# ------------------- Tab 1: Indexing ------------------- #
with tab1:
    st.header("Index Your Documents")
    uploaded_files = st.file_uploader("Upload PDF Reports", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process Documents"):
        logging.info(f"Processing {len(uploaded_files)} uploaded files.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(uploaded_files)

        new_embeddings = []

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{total_files})")
                logging.info(f"Starting processing for file: {uploaded_file.name}")
                doc_id = str(uuid.uuid4())
                images = pdf_to_images(uploaded_file)
                text = extract_text_from_pdf(uploaded_file)

                if text.strip():
                    logging.info(f"Extracting text embedding for {uploaded_file.name}")
                    emb = get_document_embedding(text, "text")
                    if emb is not None:
                        new_embeddings.append({"embedding": emb, "doc_id": doc_id, "content_type": "text"})
                        st.session_state.docs_info.append({
                            "doc_id": doc_id,
                            "source": uploaded_file.name,
                            "content_type": "text",
                            "content": text,
                            "preview": text[:200] + "..." if len(text) > 200 else text,
                        })

                for page_num, img in enumerate(images, 1):
                    page_id = f"{doc_id}_page_{page_num}"
                    logging.info(f"Extracting image embedding for page {page_num} of {uploaded_file.name}")
                    emb = get_document_embedding(img, "image")
                    if emb is not None:
                        new_embeddings.append({"embedding": emb, "doc_id": page_id, "content_type": "image"})
                        path = save_image_preview(img, f"{page_id}.png")
                        st.session_state.docs_info.append({
                            "doc_id": page_id,
                            "source": uploaded_file.name,
                            "content_type": "image",
                            "page": page_num,
                            "preview": path,
                        })

                progress_bar.progress((i + 1) / total_files)

            except Exception as e:
                logging.error(f"Error processing {uploaded_file.name}: {e}")

        save_embeddings_and_info(new_embeddings, st.session_state.docs_info)
        logging.info("Saved new embeddings and docs_info.")
        st.session_state.faiss_index, _ = load_embeddings_and_info()  # Reload after save
        logging.info("Reloaded FAISS index after saving.")
        st.success("All documents processed and indexed!")

# ------------------- Tab 2: Search ------------------- #
with tab2:
    st.header("Ask a Question")
    query = st.text_input("Enter your query (e.g., What is the profit of Visa?)")

    if query:
        logging.info(f"Received search query: {query}")
        if st.session_state.faiss_index is None:
            st.warning("No documents indexed yet.")
            logging.warning("Search attempted with no documents indexed.")
        else:
            logging.info("Starting document search.")
            results = search_documents(query, st.session_state.faiss_index, st.session_state.docs_info, get_query_embedding, top_k=3)
            logging.info(f"Search completed. Found {len(results)} results.")
            if not results:
                st.warning("No relevant results found.")
                logging.info("No relevant search results found.")
            else:
                text_result = next((r for r in results if r['content_type'] == 'text'), None)
                image_result = next((r for r in results if r['content_type'] == 'image'), None)

                with st.spinner("Generating LLM answer..."):
                    logging.info("Generating LLM answer.")
                    if image_result:
                        content = Image.open(image_result['preview'])
                        logging.info(f"Using image result for LLM: {image_result['preview']}")
                    elif text_result:
                        content = text_result['content']
                        logging.info("Using text result for LLM.")
                    else:
                        content = ""
                        logging.info("No relevant content found for LLM.")

                    answer = answer_with_gemini(query, content)
                    logging.info("LLM answer generated.")
                    st.markdown(f"### 🤖 LLM Answer:\n**{answer}**")

                if image_result:
                    st.subheader(f"🖼️ Image Match: Page {image_result['page']} from {image_result['source']}")
                    img = Image.open(image_result['preview'])
                    st.image(img, caption=None, width=1000)
                    logging.info("Displayed image match.")

# ------------------- Sidebar ------------------- #
with st.sidebar:
    st.header("Index Stats")
    if st.session_state.faiss_index is not None and st.session_state.docs_info:
        st.write(f"Total indexed items: {len(st.session_state.docs_info)}")
        logging.info(f"Displaying index stats: {len(st.session_state.docs_info)} items indexed.")
        content_types = [doc["content_type"] for doc in st.session_state.docs_info]
        type_counts = pd.Series(content_types).value_counts()

        fig, ax = plt.subplots()
        ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)

        if st.button("Clear All Indexed Data"):
            logging.info("Clearing all indexed data.")
            st.session_state.docs_info = []
            os.remove("data/faiss.index")
            os.remove("data/docs_info.pkl")
            st.success("Cleared all indexed data.")
            logging.info("Indexed data cleared successfully.")
            st.experimental_rerun()
    else:
        st.write("No documents indexed yet.")
        logging.info("No documents indexed yet.")