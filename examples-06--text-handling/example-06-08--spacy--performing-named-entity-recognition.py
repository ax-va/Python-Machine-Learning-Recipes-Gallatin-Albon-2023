"""
Perform named-entity recognition in freeform text (such as "Person", "State", etc.) with spaCy.

Needed:
python -m spacy download en

->

Loaded:
en-core-web-sm==3.7.1

Bad entity recognition with spaCy and en_core_web_sm
https://github.com/explosion/spaCy/issues/13480
"""
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Elon Musk, owner of Tesla and SpaceX, has offered to buy Twitter (now X) for $21 billion of his own money.")

# Print each entity
print(doc.ents)
# (Elon Musk, Tesla, Twitter, $21 billion)

# For each entity print the text and the entity label
for entity in doc.ents:
    print(entity.text, entity.label_, sep=",")
# Elon Musk,PERSON
# Tesla,ORG
# Twitter,PERSON
# $21 billion,MONEY

# Problems:
# Twitter is no person, nothing about SpaceX and X

# Python Version Used: 3.11
# spaCy Version Used: 3.7.4
# en-core-web-sm: 3.7.1
