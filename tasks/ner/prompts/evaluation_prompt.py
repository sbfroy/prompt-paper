EVALUATION_PROMPT = """Du er en ekspert på Named Entity Recognition (NER). 
Din oppgave er å identifisere entiteter som representerer feltnavn i tekstutdrag fra reguleringsplaner.

De eneste gyldige etikettene er B-FELT (begynnelsen på et feltnavn) og I-FELT (fortsettelsen av det samme feltnavnet).

Her er noen eksempler:

{examples}

Formuler svaret over flere linjer, med ett token per linje, og kun tokens som inngår i ett feltnavn. Hver linje skal inneholde tokenet etterfulgt av tilhørende etikett, atskilt med ett mellomrom.

Tekst: '{test_sentence}'

Entiteter:
"""
