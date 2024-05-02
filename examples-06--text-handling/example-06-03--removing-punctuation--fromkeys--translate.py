"""
Demonstrate a very fast, hacky solution to delete punctuation characters in text with
- fromkeys
- translate
"""
import unicodedata
import sys

text_data = [
    'Hi!!!! I. Love. This. Song....',
    '10000% Agree!!!! #LoveIT',
    'Right?!?!',
]
# ['Hi!!!! I. Love. This. Song....', '10000% Agree!!!! #LoveIT', 'Right?!?!']

# Create a dictionary of punctuation characters
punctuation = dict.fromkeys(
    (
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
    ),
    None
)
# {33: None,
#  34: None,
#  35: None,
#  37: None,
# ...
#  121482: None,
#  121483: None,
#  125278: None,
#  125279: None}

sys.maxunicode
# 1114111

for i in [33, 34, 35, 37]:
    print(repr(chr(i)))
# '!'
# '"'
# '#'
# '%'

for i in [33, 34, 35, 37]:
    print(unicodedata.category(chr(i)))
# Po
# Po
# Po
# Po

# For each string, remove any punctuation characters
[string.translate(punctuation) for string in text_data]
# ['Hi I Love This Song', '10000 Agree LoveIT', 'Right']
