from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

class Config:
    train_size = 1000
    test_size = 400
    epochs = 3
    learning_rate = 2e-5

conf = Config()

print("Loading 20newsgroups dataset")
dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
documents = dataset.data
labels = dataset.target
num_classes = len(dataset.target_names)

print(f"Total documents: {len(documents)}")
print(f"Number of classes: {num_classes}")
print(f"Classes: {dataset.target_names[:5]}...")

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    documents, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")

# pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# custom dataset
class NewsGroupDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_dataset = NewsGroupDataset(train_texts[:conf.train_size], train_labels[:conf.train_size], tokenizer)
test_dataset = NewsGroupDataset(test_texts[:conf.test_size], test_labels[:conf.test_size], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
total_steps = len(train_loader) * conf.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    return predictions, true_labels

print("\nTraining BERT Model")
for epoch in range(conf.epochs):
    print(f"\nEpoch {epoch + 1}/{conf.epochs}")
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

print("\nTesting BERT Model")
predictions, true_labels = eval_model(model, test_loader, device)

test_accuracy = accuracy_score(true_labels, predictions)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=[dataset.target_names[i] for i in sorted(set(test_labels[:conf.test_size]))]))

model.save_pretrained('./bert_20newsgroups_model')
tokenizer.save_pretrained('./bert_20newsgroups_model')
print("\nModel saved to './bert_20newsgroups_model'")
