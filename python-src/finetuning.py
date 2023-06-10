import os
import json
import torch
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm, trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PeftModel,
    PeftConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
    TaskType
)
import transformers.utils
import datasets.utils
from datasets import (
    load_dataset,
    concatenate_datasets,
    DatasetDict
)
import evaluate

model_name_or_path = "microsoft/deberta-v3-base"#"microsoft/deberta-base"
#task = "mrpc"
num_epochs = 20
lr = 1e-3
batch_size = 16

shuffle_seed = 17
configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

def process_level(example):
    level = example['level']
    if level == 'Level ?':
        example['level'] = None
        return example
    # Extract the number from the level and subtract 1 to get range 0-4
    level_number = int(level.split()[-1]) - 1
    assert(0 <= level_number <= 4)
    example['level'] = level_number
    return example

def load_split_dataset(dataset_name : str, split : str):
    datasets = []
    for config in configs:
        data = load_dataset(dataset_name, config)
        datasets.append(data[split])
    ds = concatenate_datasets(datasets)
    return ds.map(process_level).filter(lambda x:x['level'] != None).shuffle(shuffle_seed)

def save_combined_dataset(dataset_name : str, path : str):
    train_dataset = load_split_dataset(dataset_name, "train")
    test_dataset = load_split_dataset(dataset_name, "test")

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Save your dataset to disk
    dataset_dict.save_to_disk(path)

#save_combined_dataset("/root/rdgr/data/MATH/", "/root/rdgr/data/MATH-combined/")

dataset = DatasetDict.load_from_disk("/root/rdgr/data/MATH-combined/")
#dataset = load_dataset("/root/rdgr/data/MATH", "algebra").map(process_level).filter(lambda x:x['level'] != None)
#dataset = load_dataset("glue", "mrpc")
print(dataset["train"][0])

metric = evaluate.load('mse')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(references=labels, predictions=predictions)

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    outputs = tokenizer(examples["problem"], truncation=True, max_length=480)
    #outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    return outputs

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    #remove_columns=["idx", "sentence1", "sentence2"],
    remove_columns=["problem", "solution", "type"],
)

tokenized_datasets = tokenized_datasets.rename_column("level", "labels")
#tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
print(tokenized_datasets["train"][0])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

peft_config = PromptTuningConfig(task_type="SEQ_CLS",
                                 num_virtual_tokens=20,
                                 prompt_tuning_init="TEXT",
                                 prompt_tuning_init_text="Determine the difficulty (0-4) of the following high school math problem: ",
                                 tokenizer_name_or_path="microsoft/deberta-base"
                                 )

#peft_config = LoraConfig(
#    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1
#)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=5
                                                           )
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="joshcho/deberta-base-peft-ptuning",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = 5#training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

def train_model():
    trainer.train()
    model.save_pretrained("./saved-peft")

train_model()

def run_inference(peft_model_path):
    config = PeftConfig.from_pretrained(peft_model_path)
    inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, return_dict=True, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(inference_model, peft_model_path)

    classes = [0, 1, 2, 3, 4]
    inputs = tokenizer("Please simplify the following expression: a + 2 + 3", truncation=True, padding="longest", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
        print(outputs)

    paraphrased_text = torch.softmax(outputs, dim=1).tolist()[0]
    for i in range(len(classes)):
        print(f"{classes[i]}: {int(round(paraphrased_text[i] * 100))}%")
#run_inference()

# mse_score = mse_metric.compute(y_true, y_pred)


# def load_math_data(data_dir):
#     math_data = []
#     for subdir, _, files in os.walk(data_dir):
#         for file in files:
#             if file.endswith(".json"):
#                 file_path = os.path.join(subdir, file)
#                 with open(file_path, 'r') as f:
#                     data = json.load(f)
#                     math_data.append(data)
#     df = pd.DataFrame(math_data)
#     return df

# data_dir = "/root/rdgr/data/MATH/train"
# df = load_math_data(data_dir)

# class DifficultyDataset(Dataset):
#     def __init__(self, df, tokenizer, max_length):
#         self.df = df[df['level'] != 'Level ?']
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         problem = self.df.iloc[idx]['problem']
#         # print(self.df.iloc[idx]['level'])
#         label = int(self.df.iloc[idx]['level'][-1]) - 1  # assuming 'level' is the label
#         assert 0 <= label <= 4
#         encoding = self.tokenizer.encode_plus(
#             problem,
#             truncation=True,
#             max_length=self.max_length,
#             padding="max_length",
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
#         return {
#             'problem_text': problem,
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")

# dataset_train = DifficultyDataset(df, tokenizer, max_length=256)
# dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)

# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1
# )

# model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge", num_labels=5)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# optimizer = AdamW(model.parameters(), lr=1e-5)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)

# def train_epoch(model, dataloader, optimizer, device):
#     model.train()
#     total_loss = 0
#     progress_bar = trange(len(dataloader), desc='Training')

#     for i, batch in enumerate(dataloader):
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)

#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         total_loss += loss.item()

#         loss.backward()
#         optimizer.step()
#         progress_bar.set_description(f"Training loss: {loss.item()}")
#         progress_bar.update()

#     avg_train_loss = total_loss / len(dataloader)
#     return avg_train_loss

# def train_model(model, dataloader, epochs, optimizer, device):
#     for epoch in range(epochs):
#         print(f"\nEpoch {epoch + 1}")
#         avg_train_loss = train_epoch(model, dataloader, optimizer, device)
#         print(f"\nAverage training loss: {avg_train_loss:.2f}")

#         checkpoint_dir = f"./saved/checkpoint_epoch_{epoch + 1}"
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         model.save_pretrained(checkpoint_dir)
#         tokenizer.save_pretrained(checkpoint_dir)
#         accuracy = evaluate_and_save(model, dataloader_test, device, df_test, f"./saved/train-results-epoch-{epoch + 1}.csv")
#         print(f"Test Accuracy: {accuracy}")
#     model.save_pretrained("./saved-model")

# # Load test data
# data_dir_test = "/root/rdgr/data/MATH/test"
# df_test = load_math_data(data_dir_test)

# # Create test DataLoader
# dataset_test = DifficultyDataset(df_test, tokenizer, max_length=256)
# dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)

# def evaluate_and_save(model, dataloader, device, df, filename):
#     model.eval()
#     total = 0
#     correct = 0
#     predictions = []
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids, attention_mask=attention_mask)
#             _, preds = torch.max(outputs.logits, dim=1)
#             total += labels.size(0)
#             correct += (preds == labels).cpu().numpy().sum()

#             # Converting predictions from tensor to list
#             preds = preds.cpu().tolist()

#             # Converting label indices to 'Level n' format and appending to the list
#             predictions.extend(['Level ' + str(i+1) for i in preds])

#     accuracy = correct / total

#     # Append predictions to the DataFrame
#     df['prediction'] = predictions

#     # Save DataFrame to a CSV file
#     df.to_csv(filename, index=False)

#     return accuracy

# # Call the function to train the model
# train_model(model, dataloader_train, 10, optimizer, device)

# # Call the function to evaluate the model and save results
# train_accuracy = evaluate_and_save(model, dataloader_train, device, df_test, './train-results.csv')
# print(f"Train Accuracy: {train_accuracy}")
# test_accuracy = evaluate_and_save(model, dataloader_test, device, df_test, './test-results.csv')
# print(f"Test Accuracy: {test_accuracy}")
