"""Generate embeddings for dataset examples using OpenAI-compatible API."""

import os

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from tqdm import tqdm

from grasp.schemas import EmbeddedExample, EmbeddedDataset, InputDataset

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings using a local embedding model server."""

    def __init__(self):
        """Initialize the embedding model client.
        
        Connects to a local embedding server using environment variables:
            - EMBEDD_PORT: Server port (default: 8001)
            - EMBEDD_API_KEY: API key (default: "prompt-paper")
            - EMBEDD_MODEL: Model name (default: "Qwen/Qwen3-Embedding-4B")
        """
        self.model = OpenAIEmbeddings(
            base_url=f'http://localhost:{os.getenv("EMBEDD_PORT", "8001")}/v1',
            api_key=os.getenv("EMBEDD_API_KEY", "prompt-paper"),
            model=os.getenv("EMBEDD_MODEL", "Qwen/Qwen3-Embedding-4B"),
            tiktoken_enabled=True
        )

    def generate_embeddings(
        self, input_dataset: InputDataset, batch_size: int
    ) -> EmbeddedDataset:
        """Generate embeddings for all examples in the dataset.
        
        Args:
            input_dataset: Dataset containing input/output examples
            batch_size: Number of examples to process per batch
        
        Returns:
            EmbeddedDataset with embeddings for each example
        """
        embedded_examples = []

        for i in tqdm(
            range(0, len(input_dataset.examples), batch_size),
            desc="Generating embeddings",
            ncols=75,
        ):
            batch = input_dataset.examples[i:i + batch_size]
            
            # Merge input and output into a single text for embedding
            texts = [self._merge_input_output(ex.input, ex.output) for ex in batch]

            embeddings = self._get_embeddings(texts)

            for ex, emb in zip(batch, embeddings):
                embedded_examples.append(
                    EmbeddedExample(
                        id=ex.id,
                        input=ex.input,
                        output=ex.output,
                        embedding=emb
                    )
                )

        return EmbeddedDataset(
            examples=embedded_examples,
            task_type=input_dataset.task_type
        )

    def _merge_input_output(self, input_text: str, output_text: str) -> str:
        """Merge input and output into a single string for embedding.
        
        Args:
            input_text: The input text from the example
            output_text: The output text from the example
        
        Returns:
            Formatted string combining input and output
        """
        return f"Input: {input_text}\nOutput: {output_text}"

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from the model for a list of texts.
        
        Args:
            texts: List of strings to embed
        
        Returns:
            List of embedding vectors
        """
        return self.model.embed_documents(texts)

