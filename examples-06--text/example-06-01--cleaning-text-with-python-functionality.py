"""
Look at Python's functionality for working with text.
"""
import re

# Create text
text_data = [
    "    Interrobang. By Aishwarya Henriette        ",
    "Parking And Going. By Karl Gautier",
    "Today Is The night.      By Jarek Prakash",
]
# ['    Interrobang. By Aishwarya Henriette        ',
#  'Parking And Going. By Karl Gautier',
#  'Today Is The night.      By Jarek Prakash']

# Strip whitespaces
strip_whitespace = [string.strip() for string in text_data]
# ['Interrobang. By Aishwarya Henriette',
#  'Parking And Going. By Karl Gautier',
#  'Today Is The night.      By Jarek Prakash']

# Remove periods
remove_periods = [string.replace(".", "") for string in strip_whitespace]
# ['Interrobang By Aishwarya Henriette',
#  'Parking And Going By Karl Gautier',
#  'Today Is The night      By Jarek Prakash']


# Create and apply a custom transformation function
def capitalize(string: str) -> str:
    return string.upper()


[capitalize(string) for string in remove_periods]
# ['INTERROBANG BY AISHWARYA HENRIETTE',
#  'PARKING AND GOING BY KARL GAUTIER',
#  'TODAY IS THE NIGHT      BY JAREK PRAKASH']


def replace_letters(string: str, with_str) -> str:
    return re.sub(r"[a-zA-Z]", with_str, string)


[replace_letters(string, "XY") for string in remove_periods]
# ['XYXYXYXYXYXYXYXYXYXYXY XYXY XYXYXYXYXYXYXYXYXY XYXYXYXYXYXYXYXYXY',
#  'XYXYXYXYXYXYXY XYXYXY XYXYXYXYXY XYXY XYXYXYXY XYXYXYXYXYXYXY',
#  'XYXYXYXYXY XYXY XYXYXY XYXYXYXYXY      XYXY XYXYXYXYXY XYXYXYXYXYXYXY']

# # # more examples

# Define a string
s = "machine learning in python"
# 'machine learning in python'

# Find the first index of the letter "n"
find_n = s.find("n")
# 5

# Whether the string starts with "m"
starts_with_m = s.startswith("m")
# True

# Whether the string ends with "python"
ends_with_python = s.endswith("python")
# True

# Is the string alphanumeric
is_alnum = s.isalnum()
# False

# Is it composed of only alphabetical characters (not including spaces)
is_alpha = s.isalpha()
# False

# Encode as utf-8
encode_as_utf8 = s.encode("utf-8")
# b'machine learning in python'

# Decode the same utf-8
decode = encode_as_utf8.decode("utf-8")
# 'machine learning in python'
