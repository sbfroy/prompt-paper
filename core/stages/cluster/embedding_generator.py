from openai import OpenAI
from dotenv import load_dotenv
from typing import List

from ...schemas import InputDataset, EmbeddedExample, EmbeddedDataset

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def generate_embeddings(self, input_dataset: InputDataset) -> EmbeddedDataset:
        embedded_examples: List[EmbeddedExample] = []
        
        for example in input_dataset.examples:
            embedding = self._get_embedding(example.text)
            embedded_example = EmbeddedExample(
                example_id=example.example_id,
                text=example.text,
                embedding=embedding
            )
            embedded_examples.append(embedded_example)
                
        return EmbeddedDataset(
            examples=embedded_examples,
            task_type=input_dataset.task_type
        )

    def _get_embedding(self, text: str):
        text = text.replace("\n", " ")

        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )

        return response.data[0].embedding