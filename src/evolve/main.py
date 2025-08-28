from pathlib import Path
from deap import tools

from .experiment import GAConfig, GA
from .operators import mate, mutate
from .evaluations import ner_evaluation

if __name__ == "__main__":
    
    # get config and such

    #TODO: Need to rerun the embedding + clustering so the validation examples are separated

    ga = GA()
    best, logbook = ga.run()

# TODO: 1. fix client make sure it returns a json
# TODO: 2. implement evaluation
# TODO: 3. set up mutation functions
# TODO: 4. implement wandb
# TODO: 5. set up something that tracks cost
