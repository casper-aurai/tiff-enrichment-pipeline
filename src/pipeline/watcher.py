import time
import logging
from pipeline.main import TIFFPipelineMain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Watcher")

def main():
    input_dir = "/data/input"
    output_dir = "/data/output"
    interval = 30  # seconds

    pipeline = TIFFPipelineMain(input_dir, output_dir)
    logger.info("Starting file watcher daemon...")
    while True:
        logger.info("Checking for new files...")
        pipeline.run()
        time.sleep(interval)

if __name__ == "__main__":
    main() 