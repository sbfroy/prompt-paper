import logging
from src.config import config
from src.scope import get_batches

logging.basicConfig(level=logging.INFO)

batches = get_batches(config)

for batch in batches['batches']:
    logging.info(f"#### batch_id: {batch['id']} ####\n")
    logging.info(f"{batch['text']}\n")