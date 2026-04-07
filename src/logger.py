import logging
import os
from datetime import datetime

# fix quotes
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# correct folder
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# correct file path
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(lineno)d %(name)s - %(message)s',
    filename=LOG_FILE_PATH
)

