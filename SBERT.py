from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

print("Loading 20newsgroups dataset")
dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
documents = dataset.data
labels = dataset.target
num_classes = len(dataset.target_names)

print(f"Total documents: {len(documents)}")
print(f"Number of classes: {num_classes}")
print(f"Classes: {dataset.target_names[:5]}...")

# into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    documents, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")

# 3. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

train_subset_size = 2000
test_subset_size = 400

train_texts_subset = train_texts[:train_subset_size]
train_labels_subset = train_labels[:train_subset_size]
test_texts_subset = test_texts[:test_subset_size]
test_labels_subset = test_labels[:test_subset_size]

print("\n=== Creating training pairs for fine-tuning ===")
train_examples = []

num_pairs = 1000
for _ in tqdm(range(num_pairs), desc="Creating pairs"):
    idx1, idx2 = np.random.choice(len(train_texts_subset), 2, replace=False)
    while train_labels_subset[idx1] != train_labels_subset[idx2]:
        idx1, idx2 = np.random.choice(len(train_texts_subset), 2, replace=False)

    train_examples.append(InputExample(texts=[str(train_texts_subset[idx1]), str(train_texts_subset[idx2])], label=1.0))

    idx1, idx2 = np.random.choice(len(train_texts_subset), 2, replace=False)
    while train_labels_subset[idx1] == train_labels_subset[idx2]:
        idx1, idx2 = np.random.choice(len(train_texts_subset), 2, replace=False)

    train_examples.append(InputExample(texts=[str(train_texts_subset[idx1]), str(train_texts_subset[idx2])], label=0.0))

print("\nFine-tuning SBERT Model")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    show_progress_bar=True
)

print("Fine-tuning completed!")

print("\nGenerating embeddings for classification")

print("Encoding training texts...")
train_embeddings = model.encode(
    [str(text) for text in train_texts_subset],
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Encoding test texts...")
test_embeddings = model.encode(
    [str(text) for text in test_texts_subset],
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Train embeddings shape: {train_embeddings.shape}")
print(f"Test embeddings shape: {test_embeddings.shape}")

print("\nTraining Logistic Regression Classifier")
classifier = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
classifier.fit(train_embeddings, train_labels_subset)

print("\nTesting SBERT Model")
predictions = classifier.predict(test_embeddings)

test_accuracy = accuracy_score(test_labels_subset, predictions)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(
    test_labels_subset,
    predictions,
    target_names=[dataset.target_names[i] for i in sorted(set(test_labels_subset))]
))

model.save('./sbert_20newsgroups_model')
print("\nSentence-BERT model saved to './sbert_20newsgroups_model'")

print("\n=== Testing with example sentences ===")
test_sentences = [
    "NASA launched a new space mission to Mars.",
    "The hockey team won the championship game.",
    "New graphics card released with improved performance.",
]

test_sentence_embeddings = model.encode(test_sentences)
predictions = classifier.predict(test_sentence_embeddings)

for sentence, pred in zip(test_sentences, predictions):
    print(f"Text: {sentence}")
    print(f"Predicted category: {dataset.target_names[pred]}\n")
