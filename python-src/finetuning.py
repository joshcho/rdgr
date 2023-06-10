import os
import json
import torch
import pandas as pd
import numpy as np
import sys
import argparse
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

parser = argparse.ArgumentParser(description='Training script')

parser.add_argument('-m', '--model_name_or_path', type=str, default="microsoft/deberta-v3-base", help='Path to the model or model name')
parser.add_argument('-e', '--num_epochs', type=int, default=20, help='Number of epochs for training')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for training and evaluation')
parser.add_argument('-l', '--log_level', type=int, default=5, help='Log level for transformers and datasets')
args = parser.parse_args()

model_name_or_path = args.model_name_or_path
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = 1e-3

shuffle_seed = 17
configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

def process_level(example):
    level = example['level']
    if level == 'Level ?':
        example['level'] = None
        return example
    # Extract the number from the level and subtract 1 to get range 0-4
    level_number = float(int(level.split()[-1]) - 1)
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

def create_dispatch_datasets(dataset, dispatch_sizes, output_folder):
    # Create dispatch datasets for train and test
    for split in ['train', 'test']:
        for size in dispatch_sizes:
            dispatch_dataset = dataset[split].select(range(size))
            dispatch_dataset.to_csv(f"{output_folder}/dispatch-{split}-{size}.csv")

#dispatch_sizes = [5, 100, 1000]
#output_folder = "/root/rdgr/data/MATH-dispatch/"
#create_dispatch_datasets(dataset, dispatch_sizes, output_folder)

metric = evaluate.load('mse')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    #predictions = np.argmax(predictions, axis=1)
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

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=f"joshcho/{model_name_or_path}-peft-ptuning",
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
log_level = args.log_level
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

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
    inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, return_dict=True, num_labels=1)
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
