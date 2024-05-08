"""
Classify the sentiment of some text to use as a feature or in downstream data analysis with transformers.

See also:
https://huggingface.co/docs/transformers/quicktour
"""
from transformers import pipeline  # TensorFlow or PyTorch (named torch for Python) needed

# Create an NLP pipeline that runs sentiment analysis
classifier = pipeline("sentiment-analysis")

# Classify some text (this may download some data and models the first time you run it)
sentiment_1 = classifier("I hate machine learning! It's the absolute worst. No! No! No!")
sentiment_2 = classifier("I hate machine learning! It's the absolute worst.")
sentiment_3 = classifier("I love machine learning! It's the absolute best.")
sentiment_4 = classifier("Machine learning is the absolute bees knees I love it so much! Great! Excellent!")

# Print sentiment outputs
print("sentiment_1:\n", sentiment_1)
# sentiment_1:
#  [{'label': 'NEGATIVE', 'score': 0.999788224697113}]
print("sentiment_2:\n", sentiment_2)
# sentiment_2:
#  [{'label': 'NEGATIVE', 'score': 0.9998020529747009}]
print("sentiment_3:\n", sentiment_3)
# sentiment_3:
#  [{'label': 'POSITIVE', 'score': 0.9998683929443359}]
print("sentiment_4:\n", sentiment_4)
# sentiment_4:
#  [{'label': 'POSITIVE', 'score': 0.9997971653938293}]
