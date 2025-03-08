import logging
import structlog
from pathlib import Path

LOG_FILE = Path("prowler.log")

# Ensure logging file exists
LOG_FILE.touch(exist_ok=True)

# Configure standard logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for verbose logging
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),  # Log to file
        logging.StreamHandler()  # Log to console
    ],
)

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

log = structlog.get_logger("prowler")
log.info("Logger initialized")  # Test log message
