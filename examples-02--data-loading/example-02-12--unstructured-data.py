import requests

txt_url = "https://machine-learning-python-cookbook.s3.amazonaws.com/text.txt"

r = requests.get(txt_url)
print(r.content)
# b'Hello there!'
