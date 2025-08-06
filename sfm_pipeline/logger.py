import logging
import sys

def setup_logger(debug: bool = False):
    """
    Configures the root logger for the application.

    Args:
        debug: If True, sets logging level to DEBUG for verbose output.
               Otherwise, sets it to INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Create a custom formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-5.5s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate messages
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a handler to print to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
