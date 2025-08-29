from openai import OpenAI
from collections import defaultdict
from src.utils import get_openai_api_key

def format_examples(example_subset):
    # Formats the examples into a string for later prompt
    formatted = []
    for i, ex in enumerate(example_subset):
        entity_lines = "\n".join([f"{e['word']} {e['label']}" for e in ex["entities"]])
        formatted.append(
            f"Eksempel {i+1}:\n"
            f"Tekst: \"{ex['sentence']}\"\n"
            f"Entiteter:\n{entity_lines}\n##\n"
        )
    return "\n".join(formatted)

def evaluate_example_subset(examples, sentence, tokens, true_labels):
    client = OpenAI(api_key=get_openai_api_key())

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": ("Du er en ekspert på Named Entity Recognition (NER). "
                        "Din oppgave er å identifisere entiteter som representerer "
                        "feltnavn i tekstutdrag fra reguleringsplaner. "
                        "Svar alltid med gyldig JSON.")
            },
            {
                "role": "user", 
                "content": f"""\
    De eneste gyldige etikettene er B-FELT (begynnelsen på et feltnavn) og I-FELT (fortsettelsen av det samme feltnavnet).

    {format_examples(examples)}

    Formuler svaret over flere linjer, med ett token per linje, og kun tokens som inngår i ett feltnavn. Hver linje skal inneholde tokenet etterfulgt av tilhørende etikett, atskilt med ett mellomrom.

    Tekst: '{sentence}'

    Entiteter:
    """
            }
        ]
    )

    entities = defaultdict(list)

    for line in response.choices[0].message.content.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            word, label = parts[0], parts[1]
            entities[word].append(label)
    
    pred_labels = []
    word_counts = defaultdict(int)  # Track occurrences of each word

    for token in tokens:
        if token in entities and word_counts[token] < len(entities[token]):
            pred_labels.append(entities[token][word_counts[token]])  # Get the label in order
            word_counts[token] += 1  # Increment occurrence counter
        else:
            pred_labels.append("O")  # Default to "O" if missing

    pred_ids = []
    for label in pred_labels:
        if label in label_to_id:
            pred_ids.append(label_to_id[label])
        else:
            pred_ids.append(label_to_id.get("O", -1))

    true_ids = [label_to_id[label] for label in true_labels]
    metrics = evaluate(true_ids, pred_ids)

    return metrics['f1'] 

    # TODO: CLUSTER ONLY TRAIN DATASET SO I CAN USE DEV AND TEST BY MYSLED