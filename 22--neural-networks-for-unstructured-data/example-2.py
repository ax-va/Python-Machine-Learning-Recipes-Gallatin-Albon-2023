from sklearn.datasets import fetch_20newsgroups

cats = ['alt.atheism', 'sci.space']
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True,
random_state=42, categories=cats)