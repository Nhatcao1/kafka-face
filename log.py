import os.path

from core.config import get_app_settings
import logging
import logging.handlers


settings = get_app_settings()

logger = logging.getLogger('my_logger')

logger.setLevel(settings.logging_level)

if not os.path.exists("logs"):
    os.makedirs("logs")
# Create a rotating file handler and set its level
file_handler = logging.handlers.RotatingFileHandler(
    'logs/warning.log', maxBytes=1024 * 1024, backupCount=5)
file_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)
