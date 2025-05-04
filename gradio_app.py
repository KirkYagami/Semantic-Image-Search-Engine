import gradio as gr
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
import constants
import logging
from pathlib import Path
import time
from datetime import datetime

from logger import get_logger, set_log_level

set_log_level(logging.INFO)
logger = get_logger(__name__)


def initialize_database():
    """Initialize and return ChromaDB client and collection."""
    try:
        # Ensure DB directory exists
        Path(constants.DB_PATH).mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path=constants.DB_PATH)
        embedding_function = OpenCLIPEmbeddingFunction()
        data_loader = ImageLoader()
        
        collection = client.get_or_create_collection(
            name=constants.COLLECTION_NAME,
            embedding_function=embedding_function,
            data_loader=data_loader
        )
        
        logger.info(f"Successfully connected to database at {constants.DB_PATH}")
        logger.info(f"Collection '{constants.COLLECTION_NAME}' contains {collection.count()} images")
        
        return collection
    
    except Exception as e:
        logger.error(f"Failed to set up database: {e}")
        return None


def format_search_result(image_paths, distances):
    """Format search results with proper metadata."""
    results = []
    for path, distance in zip(image_paths, distances):
        # just the filename without path
        filename = os.path.basename(path)
        # dictionary with image path and metadata
        results.append((path, f"{filename} (Similarity: {(1-distance/2)*100:.1f}%)"))
    return results


def search_images(query_text, num_results=5, similarity_threshold=None):
    """
    Search for images similar to the query text.
    
    Args:
        query_text: The text query to search for
        num_results: Number of results to return
        similarity_threshold: Optional threshold for similarity score
        
    Returns:
        List of tuples containing image paths and distance information
    """
    start_time = time.time()
    
    if not query_text.strip():
        return [], "Please enter a search query."
    
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=num_results,
            include=["distances", "metadatas"]
        )
        
        if not results['ids'][0]:
            return [], "No matching images found."
        
        image_ids = results['ids'][0]
        distances = results['distances'][0]
        
        # Filter results by similarity threshold...
        filtered_results = []
        for image_id, distance in zip(image_ids, distances):
            similarity = (1 - distance/2) * 100  # distance to similarity percentage
            if similarity_threshold is None or similarity >= similarity_threshold:
                image_path = os.path.join(parent_path, image_id)
                filtered_results.append((image_path, distance))
        
        if not filtered_results:
            return [], f"No images met the similarity threshold of {similarity_threshold}%."
        
        # Sort by similarity (lowest distance first)
        filtered_results.sort(key=lambda x: x[1])
        
        # Format the results for display
        result_images = format_search_result(
            [r[0] for r in filtered_results],
            [r[1] for r in filtered_results]
        )
        
        execution_time = time.time() - start_time
        status_message = f"Found {len(result_images)} results in {execution_time:.2f} seconds."
        logger.info(f"Result Images: {result_images}")
        
        return result_images, status_message
        
    except Exception as e:
        error_msg = f"Error performing search: {str(e)}"
        logger.error(error_msg)
        return [], error_msg


def clear_results():
    """Clear search results."""
    return [], "Search cleared. Enter a new query."


collection = initialize_database()

if collection is None:
    logger.critical("Failed to initialize database. Application will not function correctly.")

# Set the path to the image directory
parent_path = r"coco_val_images"

# Check if the path exists
if not os.path.exists(parent_path):
    logger.warning(f"Image directory not found: {parent_path}")

# Create theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
)


with gr.Blocks(title=constants.APP_NAME, theme=theme) as app:
    gr.Markdown(f"# {constants.APP_NAME}")
    gr.Markdown("""
    Enter a text description to find visually similar images in our database.
    The system uses CLIP embeddings to match your text with relevant images.
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g., 'person walking a dog on the beach' or 'red sports car'",
                lines=2
            )
        with gr.Column(scale=1):
            search_button = gr.Button("Search", variant="primary")
            clear_button = gr.Button("Clear")
    
    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            num_results = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Results"
            )
            similarity_threshold = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=5,
                label="Minimum Similarity (%)"
            )
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    gallery_output = gr.Gallery(
        label="Search Results",
        columns=3,
        height="500px",
        object_fit="contain"
    )
    
    # event handlers
    search_button.click(
        fn=search_images,
        inputs=[query_input, num_results, similarity_threshold],
        outputs=[gallery_output, status_output]
    )
    
    query_input.submit(
        fn=search_images,
        inputs=[query_input, num_results, similarity_threshold],
        outputs=[gallery_output, status_output]
    )
    
    clear_button.click(
        fn=clear_results,
        inputs=[],
        outputs=[gallery_output, status_output]
    )
    
    # Show app version and timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gr.Markdown(f"*Version: {constants.APP_VERSION} | Last Updated: {current_time}*")

if __name__ == "__main__":
    if collection is None:
        print("ERROR: Could not initialize database. Please check logs.")
    else:
        app.launch(share=False, inbrowser=True)