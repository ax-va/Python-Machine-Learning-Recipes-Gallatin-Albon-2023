"""
Train an image classification model using learnings from a pretrained model.
->
Use the transformers library with torchvision to fine-tune a pretrained model on your data.

*transfer learning* = using pretrained models
ViT = Google's Vision Transformer

See also:
- Hugging Face
https://huggingface.co/
"""
import numpy as np
# to load the fashion MNIST dataset
from datasets import load_dataset, load_metric
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
# to load the weights from Google's ViT model
from transformers import ViTImageProcessor, ViTForImageClassification
# to fine-tune the ViT model for a classification task on the fashion MNIST dataset
from transformers import Trainer, TrainingArguments, DefaultDataCollator

# Load the fashion mnist dataset
dataset = load_dataset("fashion_mnist")
# Load the processor from the VIT model
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# Set the labels from the dataset
labels = dataset['train'].features['label'].names
# ['T - shirt / top',
#  'Trouser',
#  'Pullover',
#  'Dress',
#  'Coat',
#  'Sandal',
#  'Shirt',
#  'Sneaker',
#  'Bag',
#  'Ankle boot']

# Load the pretrained model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

# Define the collator, normalizer, and _transforms
collate_fn = DefaultDataCollator()
normalize = Normalize(
    mean=image_processor.image_mean,
    std=image_processor.image_std,
)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (
        image_processor.size["height"],
        image_processor.size["width"],
    )
)
_transforms = Compose(
    [
        RandomResizedCrop(size),
        ToTensor(),
        normalize,
    ]
)


# Define a helper function to convert the images into RGB
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


# Load the dataset we'll use with transformations
dataset = dataset.with_transform(transforms)
# Use accuracy as our metric
metric = load_metric("accuracy")


# Define a helper function to compute metrics
def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
    )


# Set the training args
training_args = TrainingArguments(
    output_dir="fashion_mnist_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.01,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# Instantiate a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
)

# Train the model, log and save metrics
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics(split="train", metrics=train_results.metrics)
trainer.save_metrics(split="train", metrics=train_results.metrics)
trainer.save_state()

"""
Some weights of ViTForImageClassification were not initialized from the model checkpoint at 
google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
"""

# in `metric = load_metric("accuracy")`
"""
The repository for accuracy contains custom code which must be executed to correctly load the dataset. 
You can inspect the repository content at https://hf.co/datasets/accuracy.
You can avoid this prompt in future by passing the argument `trust_remote_code=True`.

Do you wish to run the custom code? [y/N]

ValueError: The repository for accuracy contains custom code which must be executed to correctly load the dataset. 
You can inspect the repository content at https://hf.co/datasets/accuracy.
Please pass the argument `trust_remote_code=True` to allow custom code to be run.
"""
