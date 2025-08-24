from src.config import config
from src.scope import get_batches

batches = get_batches(config)
print(batches)