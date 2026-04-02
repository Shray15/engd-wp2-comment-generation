import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,classification_report
import numpy as np

final_data_classification = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Ektha Thesis\Evaluation_data_Few _llama.csv")
final_data_classification["Comments"] = final_data_classification["Comments"].apply(lambda x: str(x))

seed = 42
np.random.seed(seed)
 
train_texts, test_texts, train_labels, test_labels = train_test_split(final_data_classification["Comments"], final_data_classification["Labels"], test_size=.2,stratify=final_data_classification["Labels"])

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2,stratify=train_labels)
tokenizer = tokenizer = AutoTokenizer.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large")

def tokenize_function(examples):
    return tokenizer(examples,  padding="max_length", truncation=True, max_length=512, return_tensors="pt")

train_encodings = tokenize_function(list(train_texts))
val_encodings = tokenize_function(list(val_texts))
test_encodings = tokenize_function(list(test_texts))


# Prepare the dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Use .clone().detach() to create a new tensor from existing tensors
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, list(train_labels.astype(int)))
test_dataset = CustomDataset(test_encodings, list(test_labels.astype(int)))
val_dataset = CustomDataset(val_encodings, list(val_labels.astype(int)))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    #print("Labels",labels)
    #print("Preds",preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    class_report = classification_report(labels, preds, target_names=['Real Data', 'Generated Data'], output_dict=True)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': class_report
    }


model = AutoModelForSequenceClassification.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large",num_labels=2)

training_args = TrainingArguments(
        output_dir='./results',          # output directory
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        logging_dir='./logs',
        logging_steps=100,
        gradient_accumulation_steps=2,
        learning_rate=1e-4
    )

trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset  ,       # training dataset
        eval_dataset = val_dataset,
      compute_metrics = compute_metrics
    )

    # Train the model
trainer.train()
eval_output = trainer.evaluate(test_dataset)
print(eval_output)