import os
from pathlib import Path


DEFAULT_DB_PATH = "data/image_vec"
DEFAULT_IMAGE_FOLDER_PATH = "coco_val_images"
DEFAULT_COLLECTION_NAME = "image_embeddings"


DB_PATH = os.environ.get("IMAGE_DB_PATH", DEFAULT_DB_PATH)
COLLECTION_NAME = os.environ.get("IMAGE_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
IMAGE_FOLDER_PATH = os.environ.get("IMAGE_FOLDER_PATH", DEFAULT_IMAGE_FOLDER_PATH)


Path(DB_PATH).mkdir(parents=True, exist_ok=True)



APP_NAME = "📸 Image Search Engine"
APP_VERSION = "1.0.0"

# Search settings
DEFAULT_NUM_RESULTS = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.2

# Paths
ASSETS_PATH = "assets" # did use this earlier but removed now, may add later..
LOG_PATH = "logs"

# Ensure necessary directories exist
Path(DB_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
Path(ASSETS_PATH).mkdir(parents=True, exist_ok=True)

# validation
if not os.access(os.path.dirname(DB_PATH), os.W_OK):
    raise PermissionError(f"No write permission for DB path: {DB_PATH}")

