import numpy as np
import io
import base64
from PIL import Image
from config import COHERE_API_KEY
import cohere
import logging

# Constants
MAX_PIXELS = 1568 * 1568  # Cohere image size limit

# Initialize Cohere client
co_client = cohere.ClientV2(api_key=COHERE_API_KEY)

def resize_image(pil_image):
    """Resize image if too large for embedding API"""
    logging.info("Resizing image for embedding.")
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        resized_image = pil_image.resize((new_width, new_height))
        logging.info(f"Image resized from {org_width}x{org_height} to {new_width}x{new_height}.")
        return resized_image
    logging.info("Image size is within limits, no resizing needed.")
    return pil_image

def base64_from_image(pil_image):
    """Convert PIL Image to base64 for Cohere"""
    logging.info("Converting image to base64.")
    pil_image = resize_image(pil_image)
    img_format = pil_image.format if pil_image.format else "PNG"
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format=img_format)
        img_bytes = buffer.getvalue()
    base64_string = f"data:image/{img_format.lower()};base64," + base64.b64encode(img_bytes).decode("utf-8")
    logging.info("Image converted to base64.")
    return base64_string

def get_document_embedding(content, content_type="text"):
    """Embed document (text or image)"""
    logging.info(f"Getting document embedding for content type: {content_type}")
    try:
        if content_type == "text":
            response = co_client.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                texts=[content],
            )
            logging.info("Text embedding successful.")
            return np.array(response.embeddings.float[0])
        else:
            api_input_document = {
                "content": [
                    {"type": "image", "image": base64_from_image(content)},
                ]
            }
            response = co_client.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=[api_input_document],
            )
            logging.info("Image embedding successful.")
            return np.array(response.embeddings.float[0])
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        print(f"Embedding error: {e}")
        return None

def get_query_embedding(query):
    """Embed search query"""
    logging.info(f"Getting query embedding for query: {query[:50]}...")
    try:
        response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[query],
        )
        logging.info("Query embedding successful.")
        return np.array(response.embeddings.float[0])
    except Exception as e:
        logging.error(f"Query embedding error: {e}")
        print(f"Query embedding error: {e}")
        return None