from core.stages.evolve import get_llm_response

class Evaluator:
    def __init__(self, base_dir, config, llm_instance, sampling_params):
        self.base_dir = base_dir
        self.config = config
        self.llm_instance = llm_instance
        self.sampling_params = sampling_params

    def evaluate_individual(self, individual):
        """
        Evaluate an individual on ...

        Returns:
            float: 
        """
        # Load validation data

        # TODO: Use the json mode in vLLM
        # I need to change to the OpenAI API for that, maybe not, only if it fixes the threading issue

        




