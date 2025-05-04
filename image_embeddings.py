from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import numpy as np
from tqdm import tqdm
import os
import concurrent.futures
from pathlib import Path
import argparse
from datetime import datetime
import logging
import constants
from logger import get_logger, set_log_level


def setup_database():
    """Initialize and return ChromaDB client and collection."""
    logger = get_logger(__name__)
    
    try:
        Path(constants.DB_PATH).mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path=constants.DB_PATH)
        embedding_function = OpenCLIPEmbeddingFunction()
        data_loader = ImageLoader()
        
        collection = client.get_or_create_collection(
            name=constants.COLLECTION_NAME,
            embedding_function=embedding_function,
            data_loader=data_loader,
            metadata={"description": "Image embeddings collection", "created_at": str(datetime.now())}
        )
        
        logger.info(f"Successfully connected to database at {constants.DB_PATH}")
        logger.info(f"Collection '{constants.COLLECTION_NAME}' contains {collection.count()} images")
        
        return collection
    except Exception as e:
        logger.error(f"Failed to set up database: {e}")
        raise


def process_image(image_path):
    """Process a single image and return its data for adding to collection."""
    logger = get_logger(__name__)
    
    try:
        image_id = os.path.basename(image_path)
        
        """        
        existing = collection.get(ids=[image_id], include=[])
        if existing and existing['ids']:
            logger.debug(f"Skipping {image_id} - already in database")
            return None
        """
            
        # Open and process image
        with Image.open(image_path) as img:
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Resize save memory
            if max(img.size) > 2000:
                ratio = 2000 / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                
            image_array = np.array(img)
            
        return {
            "id": image_id,
            "image": image_array,
            "metadata": {
                "filename": image_id,
                "path": str(image_path),
                "dimensions": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1
            }
        }
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def add_images_to_collection(collection, folder_path, batch_size=50, max_workers=4):
    """
    Add images from folder to collection with batching and parallel processing.
    
    Args:
        collection: ChromaDB collection object
        folder_path: Path to folder containing images
        batch_size: Number of images to process in each batch
        max_workers: Maximum number of parallel workers
    """
    logger = get_logger(__name__)
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
    image_paths = [
        str(file) for file in folder_path.glob('**/*') 
        if file.is_file() and file.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        logger.warning(f"No images found in {folder_path}")
        return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process images in batches using parallel processing..
    batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
    total_added = 0
    
    for batch_num, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_num+1}/{len(batches)}")
        
        # Process images in parallel...
        processed_images = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(
                executor.map(process_image, batch),
                total=len(batch),
                desc=f"Processing batch {batch_num+1}"
            ):
                if result:
                    processed_images.append(result)
        
        if not processed_images:
            continue
            
        ids = [img["id"] for img in processed_images]
        images = [img["image"] for img in processed_images]
        metadatas = [img["metadata"] for img in processed_images]
        
        # adding batch to collection
        try:
            collection.add(
                ids=ids,
                images=images,
                metadatas=metadatas
            )
            total_added += len(processed_images)
            logger.info(f"Added {len(processed_images)} images to collection")
        except Exception as e:
            logger.error(f"Error adding batch to collection: {e}")
    
    logger.info(f"Successfully added {total_added} images to collection")
    logger.info(f"Collection now contains {collection.count()} images")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Add images to vector database")
    parser.add_argument("--folder", type=str, default="coco_val_images", 
                        help="Path to folder containing images")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of images to process in each batch")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    args = parser.parse_args()
    

    log_level = getattr(logging, args.log_level)
    set_log_level(log_level)
    
    logger = get_logger(__name__)
    
    try:
        collection = setup_database()

        add_images_to_collection(
            collection=collection,
            folder_path=args.folder,
            batch_size=args.batch_size,
            max_workers=args.workers
        )
        
        logger.info("Image processing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    elapsed = datetime.now() - start_time
    get_logger(__name__).info(f"Total processing time: {elapsed}")