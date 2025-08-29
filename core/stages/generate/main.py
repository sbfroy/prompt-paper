from src.config import config
from src.scope import get_batches

batches = get_batches(config)

for batch in batches['batches']:
    print(f"#### batch_id: {batch['id']} ####\n")
    print(f"{batch['text']}\n")