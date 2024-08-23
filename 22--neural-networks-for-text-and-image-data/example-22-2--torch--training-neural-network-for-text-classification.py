"""
Train a neural network to classify text data.
->
Use a PyTorch neural network whose first layer is the size of your vocabulary.

*word embeddings* = vector representations of individual words
where each word is assigned to a specific index in the vector, and
the value at that location is the number of times that word
appears in a given text part.
"""
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim import Adam

NUM_EPOCHS = 1
BATCH_SIZE = 10

cats = ['alt.atheism', 'sci.space']
newsgroups_data = fetch_20newsgroups(
    subset='all',
    shuffle=True,
    random_state=42,
    categories=cats,
)
type(newsgroups_data)
# sklearn.utils._bunch.Bunch
type(newsgroups_data.data)
# list
type(newsgroups_data.target)
# numpy.ndarray
len(newsgroups_data.data)
# 1786
newsgroups_data.target.shape
# (1786,)
newsgroups_data.data[0]
# 'From: 9051467f@levels.unisa.edu.au (The Desert Brat)\nSubject: Re: Keith Schneider
# - Stealth Poster?\nOrganization: Cured, discharged\nLines: 24\n\nIn article <1pa0f4INNpit@gap.caltech.edu>,
# keith@cco.caltech.edu (Keith Allan Schneider) writes:\n\n> But really, are you threatened by the motto,
# or by the people that use it?\n\nEvery time somone writes something and says it is merely describing the norm,\n
# it is infact re-inforcing that norm upon those programmed not to think for\nthemselves.
# The motto is dangerous in itself, it tells the world that every\n*true* American is god-fearing,
# and puts down those who do not fear gods. It\ndoesn\'t need anyone to make it dangerous,
# it does a good job itself by just\nexisting on your currency.\n\n> keith\n\nThe Desert Brat\n-- \n
# John J McVey, Elc&Eltnc Eng, Whyalla, Uni S Australia,    ________\n
# 9051467f@levels.unisa.edu.au      T.S.A.K.C.            \\/Darwin o\\\n
# For replies, mail to whjjm@wh.whyalla.unisa.edu.au      /\\________/\n
# Disclaimer: Unisa hates my opinions.                       bb  bb\n
# +------------------------------------------------------+-----------------------+\n
# |"It doesn\'t make a rainbow any less beautiful that we | "God\'s name is smack  |\n
# |understand the refractive mechanisms that chance to   | for some."            |\n
# |produce it." - Jim Perry, perry@dsinc.com             |    - Alice In Chains  |\n
# +------------------------------------------------------+-----------------------+\n'
newsgroups_data.target[0]
# 0

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups_data.data, newsgroups_data.target,
    test_size=0.2,
    random_state=42,
)
type(X_train)
# list
type(y_train)
# numpy.ndarray
len(X_train)
# 1428
len(X_test)
# 358
y_train.shape
# (1428, )
y_test.shape
# (358, )

# Vectorize the text data using a bag-of-words approach such that
# each word is assigned to a specific index in the vector, and
# the value at that location is the number of times that word
# appears in a given text part.
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train).toarray()
# array([[0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        ...,
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0]])
X_train.shape
X_test = vectorizer.transform(X_test).toarray()
# array([[0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        ...,
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [1, 0, 0, ..., 0, 0, 0]])
len(vectorizer.vocabulary_)
# 25150
vectorizer.vocabulary_
# {'sandvik': 20208,
#  'newton': 16309,
#  'apple': 4120,
#  'com': 6836,
#  'kent': 13846,
#  'subject': 21978,
#  'read': 19048,
#  'rushdie': 20079,
#  '_the': 3073,
#  'satanic': 20233,
#  'verses_': 24179,
#  'organization': 16932,
#  'cookamunga': 7430,
# ...

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

X_train.shape
# torch.Size([1428, 25150])


# Define the model
class TextClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


# Instantiate the model and define the loss function and optimizer
model = TextClassifier(num_classes=len(cats))
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)
# Compile the model using torch 2.0's optimizer
model = torch.compile(model)

num_batches = len(X_train) // BATCH_SIZE
for epoch_idx in range(1, NUM_EPOCHS + 1):
    total_loss = 0.0
    for batch_idx in range(num_batches):
        # Prepare the input and target data for the current batch
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        inputs = X_train[start_idx:end_idx]
        targets = y_train[start_idx:end_idx]
        # Zero the gradients for the optimizer
        optimizer.zero_grad()
        # Forward pass through the model and compute the loss
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        # Backward pass through the model and update the parameters
        loss.backward()
        optimizer.step()
        # Update the total loss for the epoch
        total_loss += loss.item()
        print("batch_idx:", batch_idx)
        # atch_idx: 0
        # ...
        # batch_idx: 141

# Compute the accuracy on the test set for the epoch
test_outputs = model(X_test)
test_predictions = torch.argmax(test_outputs, dim=1)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print the epoch number, average loss, and test accuracy
print(
    f"Epochs: {NUM_EPOCHS};"
    f"\t Loss: {total_loss / num_batches};"
    f"\t Test Accuracy: {test_accuracy}"
)
# Epochs: 1;	 Loss: 0.17057145314108607;	 Test Accuracy: 0.9916201117318436
