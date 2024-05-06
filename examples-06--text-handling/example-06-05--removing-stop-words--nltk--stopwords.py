"""
Remove extremely common words (e.g., a, is, of, on) from tokenized text data with NLTK.
That common words themselves contain little information value.

Needed:
import nltk
nltk.download('stopwords')  # Download the set of stop words the first time
"""
from nltk.corpus import stopwords

tokenized_words = [
    'i',
    'am',
    'going',
    'to',
    'go',
    'to',
    'the',
    'store',
    'and',
    'park',
]

# Load stop words
stop_words = stopwords.words('english')

# Remove stop words
[word for word in tokenized_words if word not in stop_words]
# ['going', 'go', 'store', 'park']

# Show stop words
stop_words[:10]
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]

len(stop_words)
# 179
