# from sentence_transformers import SentenceTransformer
import os

from tqdm import tqdm
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

from proptimize.schemas import EmbeddedExample, EmbeddedDataset, InputDataset


class EmbeddingGenerator:
    def __init__(self):
        self.model = OpenAIEmbeddings(
            base_url=f'http://localhost:{os.getenv("EMBEDD_PORT", "8001")}/v1',
            api_key=os.getenv("EMBEDD_API_KEY", "prompt-paper"),
            model=os.getenv("EMBEDD_MODEL", "Qwen/Qwen3-Embedding-4B"),
            tiktoken_enabled=True,
            # api_key=os.getenv
        )

    def generate_embeddings(
        self, input_dataset: InputDataset, batch_size: int 
    ) -> EmbeddedDataset:
        embedded_examples = []

        for i in tqdm(
            range(0, len(input_dataset.examples), batch_size),
            desc="Generating embeddings",
            ncols=75,
        ):
            batch = input_dataset.examples[i : i + batch_size]
            texts = [ex.text.replace("\n", " ") for ex in batch]

            embeddings = self._get_embeddings(texts)

            for ex, emb in zip(batch, embeddings):
                embedded_examples.append(
                    EmbeddedExample(
                        example_id=ex.example_id, text=ex.text, embedding=emb
                    )
                )

        return EmbeddedDataset(
            examples=embedded_examples, task_type=input_dataset.task_type
        )

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)
