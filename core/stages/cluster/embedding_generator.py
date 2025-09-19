from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ...schemas import EmbeddedExample, EmbeddedDataset

class EmbeddingGenerator:
    def __init__(self, model):
        self.model = SentenceTransformer(model)

    def generate_embeddings(self, input_dataset, batch_size):
        embedded_examples = []

        for i in tqdm(
            range(0, len(input_dataset.examples), batch_size), 
            desc="Generating embeddings", 
            ncols=75
        ):
            batch = input_dataset.examples[i : i + batch_size]
            texts = [ex.text.replace("\n", " ") for ex in batch]

            embeddings = self._get_embeddings(texts)

            for ex, emb in zip(batch, embeddings):
                embedded_examples.append(
                    EmbeddedExample(
                        example_id=ex.example_id,
                        text=ex.text,
                        embedding=emb
                    )
                )

        return EmbeddedDataset(
            examples=embedded_examples,
            task_type=input_dataset.task_type
        )

    def _get_embeddings(self, texts):
        return self.model.encode(texts).tolist()