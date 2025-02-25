import logging
import structlog
from pathlib import Path

LOG_FILE = Path("prowler.log")

structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.WriteLoggerFactory(
        file=LOG_FILE.open("wt")
    ),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = structlog.get_logger("prowler")
