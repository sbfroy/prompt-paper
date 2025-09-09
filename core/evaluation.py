from abc import ABC, abstractmethod

class TaskEvaluator(ABC):
    """
    Abstract Base Class (ABC) for task-specific evaluators.
    ABC makes it so that subclasses can't create instances directly.
    """
    # abstractmethod means that subclasses must implement the method
    @abstractmethod
    def evaluate_individual(self, individual, **kwargs):
        """
        Evaluate an individual and return fitness scores.

        Args:
            individual: The individual to evaluate.
            **kwargs: Additional arguments (flexibility for tasks that need different inputs)
        Returns:
            a tuple of fitness scores (float,)
        """
        pass
