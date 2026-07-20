from solarpinn.utils.logger import get_logger

logger = get_logger(__name__)

logger.debug("Debug message")
logger.info("Downloading magnetograms...")
logger.warning("Missing observation.")
logger.error("Failed to read FITS file.")
logger.critical("Unexpected fatal error.")

