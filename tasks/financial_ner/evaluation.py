from core.stages.evolve import get_llm_response

class Evaluator:
    def __init__(self, base_dir, config, client):
        self.base_dir = base_dir
        self.config = config
        self.client = client

    def evaluate_individual(self, individual):
        """
        Evaluate an individual on ...

        Returns:
            float: 
        """
        # Load validation data

        # TODO: Use json mode 




