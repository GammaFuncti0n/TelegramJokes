import yaml
import logging
from TelegramJokes import setup_logging, set_seed, check_paths
from TelegramJokes import TrainRunner

setup_logging()
logger = logging.getLogger(__name__)

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logger.info(config)
    check_paths(config)

    set_seed(config['env']['seed'])
    logger.info("Set seed: %d", config['env']['seed'])

    runner = TrainRunner(config)
    runner.run()

if __name__ == "__main__":
    main()