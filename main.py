import yaml
import logging
from TelegramJokes import setup_logging, set_seed

setup_logging()
logger = logging.getLogger(__name__)

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['env']['seed'])
    logger.info("Set seed: %d", config['env']['seed'])
    logger.info(config)

if __name__ == "__main__":
    main()