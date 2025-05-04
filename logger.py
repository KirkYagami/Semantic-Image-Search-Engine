import logging
import sys
from pathlib import Path
from datetime import datetime
import os


class Logger:
    """
    Singleton logger class for centralized logging throughout the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            # Initialize the logger on first creation
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger instance with file and console handlers."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"image_embeddings_{timestamp}.log"
        
        # Set up the root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers (important for singleton pattern)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def get_logger(self, name=None):
        """
        Get a logger instance with the specified name.
        
        Args:
            name (str, optional): Logger name, typically the module name. 
                                 Defaults to None for root logger.
        
        Returns:
            logging.Logger: Logger instance
        """
        if name:
            return logging.getLogger(name)
        return self.logger
    
    def set_level(self, level):
        """Set the logging level for all handlers."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
        self.logger.info(f"Log level set to {logging.getLevelName(level)}")


def get_logger(name=None):
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str, optional): Logger name, typically the module name.
                             Defaults to None for root logger.
    
    Returns:
        logging.Logger: Logger instance
    """
    return Logger().get_logger(name)

def set_log_level(level):
    """
    Set the global logging level.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    Logger().set_level(level)

# Example usage in this module
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    set_log_level(logging.DEBUG)
    logger.debug("This debug message should now be visible")