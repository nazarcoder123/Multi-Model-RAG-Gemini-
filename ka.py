# import streamlit as st
# import os
# import base64
# from io import BytesIO
# from PIL import Image
# import tempfile
# import fitz  # PyMuPDF for PDF processing
# import docx  # python-docx for DOCX processing
# from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema.document import Document
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_core.runnables import RunnablePassthrough

# # Load environment variables from .env file
# load_dotenv()

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Multimodal RAG Chatbot",
#     page_icon="🤖",
#     layout="wide"
# )

# st.title("🤖 Multimodal RAG Chatbot")
# st.markdown("Upload documents and chat with text and images using Google's Gemini models!")

# # Sidebar for configuration
# with st.sidebar:
#     st.header("⚙️ Configuration")

#     # Get API key from environment variable
#     google_api_key = os.getenv("GOOGLE_API_KEY")

#     if not google_api_key:
#         st.error("GOOGLE_API_KEY not found in environment variables or .env file!")

#     st.header("📁 Document Upload")
#     uploaded_files = st.file_uploader(
#         "Choose files",
#         type=['txt', 'pdf', 'docx'],
#         accept_multiple_files=True,
#         help="Upload PDF, DOCX, or TXT files"
#     )

#     # Processing options
#     st.header("🔧 Processing Options")
#     chunk_size = st.slider("Chunk Size", 100, 2000, 500)
#     chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)

# # Helper functions
# @st.cache_resource
# def load_models():
#     """Load and cache the language models"""
#     try:
#         text_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash") # gemini-2.0-flash-lite (Text)
#         vision_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")     # gemini-1.5-pro (Multi Model)
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
#         return text_model, vision_model, embeddings
#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         return None, None, None

# def extract_text_from_pdf(pdf_file):
#     """Extract text from PDF file"""
#     try:
#         # Save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(pdf_file.getvalue())
#             tmp_path = tmp_file.name

#         # Extract text using PyMuPDF
#         doc = fitz.open(tmp_path)
#         text = ""
#         for page in doc:
#             text += page.get_text()
#         doc.close()

#         # Clean up temporary file
#         os.unlink(tmp_path)
#         return text
#     except Exception as e:
#         st.error(f"Error extracting text from PDF: {str(e)}")
#         return ""

# def extract_text_from_docx(docx_file):
#     """Extract text from DOCX file"""
#     try:
#         # Save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
#             tmp_file.write(docx_file.getvalue())
#             tmp_path = tmp_file.name

#         # Extract text using python-docx
#         doc = docx.Document(tmp_path)
#         text = ""
#         for paragraph in doc.paragraphs:
#             text += paragraph.text + "\n"

#         # Clean up temporary file
#         os.unlink(tmp_path)
#         return text
#     except Exception as e:
#         st.error(f"Error extracting text from DOCX: {str(e)}")
#         return ""

# def process_uploaded_files(files, chunk_size, chunk_overlap):
#     """Process uploaded files and create documents"""
#     all_text = ""

#     for file in files:
#         if file.type == "text/plain":
#             # Handle TXT files
#             text = str(file.read(), "utf-8")
#             all_text += f"\n\n--- {file.name} ---\n\n{text}"
#         elif file.type == "application/pdf":
#             # Handle PDF files
#             text = extract_text_from_pdf(file)
#             all_text += f"\n\n--- {file.name} ---\n\n{text}"
#         elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#             # Handle DOCX files
#             text = extract_text_from_docx(file)
#             all_text += f"\n\n--- {file.name} ---\n\n{text}"

#     if not all_text.strip():
#         return None

#     # Split text into chunks
#     text_splitter = CharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )

#     docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(all_text)]
    
#     return docs

# def create_rag_chain(vectorstore, text_model):
#     """Create RAG chain from vectorstore and text model"""
#     retriever = vectorstore.as_retriever()
    
#     template = """
# Eres un experto en Roblox Studio. Responde a la siguiente consulta en español, utilizando la información de los documentos proporcionados y la descripción de la imagen si está disponible.

# Contexto de los documentos:
# ```
# {context}
# ```

# Consulta del usuario (incluye descripción de la imagen si aplica): {query}

# Basándote en el contexto y la descripción de la imagen, proporciona una respuesta completa en español. Si la información no se encuentra en los documentos o la imagen, por favor di "Pregúntame algo relacionado con el tema".
# """

#     prompt_template = ChatPromptTemplate.from_template(template)

#     rag_chain = (
#         {"context": retriever, "query": RunnablePassthrough()}
#         | prompt_template
#         | text_model
#         | StrOutputParser()
#     )

#     return rag_chain

# def pil_image_to_base64(image):
#     """Convert PIL image to base64 string"""
#     buffered = BytesIO()
#     image.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     return f"data:image/png;base64,{img_str}"

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None
# if "rag_chain" not in st.session_state:
#     st.session_state.rag_chain = None

# # Main content area
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.header("💬 Chat")

#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             if message["type"] == "text":
#                 st.markdown(message["content"])
#             elif message["type"] == "image":
#                 st.image(message["content"], width=300)

# with col2:
#     st.header("📊 Document Status")

#     # Process uploaded files
#     if uploaded_files and google_api_key:
#         with st.spinner("Processing documents..."):
#             docs = process_uploaded_files(uploaded_files, chunk_size, chunk_overlap)

#             if docs:
#                 # Load models
#                 text_model, vision_model, embeddings = load_models()

#                 if embeddings and text_model:
#                     # Create vector store
#                     vectorstore = FAISS.from_documents(docs, embedding=embeddings)
                    
#                     # Create RAG chain
#                     rag_chain = create_rag_chain(vectorstore, text_model)

#                     st.session_state.vectorstore = vectorstore
#                     st.session_state.rag_chain = rag_chain
#                     st.session_state.text_model = text_model
#                     st.session_state.vision_model = vision_model

#                     st.success(f"✅ Processed {len(docs)} document chunks")
#                     st.info(f"📄 Files: {', '.join([f.name for f in uploaded_files])}")
#                 else:
#                     st.error("❌ Failed to load models")
#             else:
#                 st.error("❌ No text could be extracted from the uploaded files")

#     elif uploaded_files and not google_api_key:
#          st.warning("⚠️ GOOGLE_API_KEY is not set in your environment or .env file.")

#     elif not uploaded_files:
#         st.info("📁 Upload documents to get started")

# # Chat input
# if google_api_key:
#     # Image upload for multimodal chat
#     uploaded_image = st.file_uploader(
#         "Upload an image for multimodal chat (optional)",
#         type=['png', 'jpg', 'jpeg'],
#         key="chat_image"
#     )

#     # Text input
#     if prompt := st.chat_input("Ask me anything about your documents or describe an image..."):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})

#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Handle image if uploaded
#         if uploaded_image:
#             image = Image.open(uploaded_image)
#             st.session_state.messages.append({"role": "user", "type": "image", "content": image})

#             with st.chat_message("user"):
#                 st.image(image, width=300)

#         # Generate response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     if uploaded_image and hasattr(st.session_state, 'vision_model'):
#                         # Multimodal response with image
#                         image_base64 = pil_image_to_base64(image)

#                         message = HumanMessage(
#                             content=[
#                                 {"type": "text", "text": prompt},
#                                 {"type": "image_url", "image_url": {"url": image_base64}}
#                             ]
#                         )

#                         # Get vision model response
#                         vision_response = st.session_state.vision_model.invoke([message]).content

#                         # If we have a RAG chain, also get document-based response
#                         if st.session_state.rag_chain:
#                             combined_query = f"{prompt}\n\nImage description: {vision_response}"
#                             rag_response = st.session_state.rag_chain.invoke(combined_query)
#                             response = f"**Image Analysis:**\n{vision_response}\n\n**Document Information:**\n{rag_response}"
#                         else:
#                             response = vision_response

#                     elif st.session_state.rag_chain:
#                         # Text-only response using RAG
#                         response = st.session_state.rag_chain.invoke(prompt)

#                     elif hasattr(st.session_state, 'text_model'):
#                         # Fallback to basic text model
#                         response = st.session_state.text_model.invoke(prompt).content

#                     else:
#                         response = "Please upload documents and ensure your API key is set to start chatting!"

#                     st.markdown(response)

#                     # Add assistant response to chat history
#                     st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})

#                 except Exception as e:
#                     error_msg = f"Error: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "type": "text", "content": error_msg})

# else:
#     st.info("🔑 Please set your Google API Key in your .env file to start chatting")