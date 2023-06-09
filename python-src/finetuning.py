import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, AdamW
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def load_math_data(data_dir):
    math_data = []
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    math_data.append(data)
    df = pd.DataFrame(math_data)
    return df

data_dir = "/root/rdgr/data/MATH/train"
df = load_math_data(data_dir)

class DifficultyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df[df['level'] != 'Level ?']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        problem = self.df.iloc[idx]['problem']
        # print(self.df.iloc[idx]['level'])
        label = int(self.df.iloc[idx]['level'][-1]) - 1  # assuming 'level' is the label
        assert 0 <= label <= 4
        encoding = self.tokenizer.encode_plus(
            problem,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'problem_text': problem,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")

dataset = DifficultyDataset(df, tokenizer, max_length=256)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge", num_labels=5)
model = get_peft_model(model, peft_config)
optimizer = AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    progress_bar = tqdm(dataloader, desc="Training", position=0, leave=True)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Training loss: {loss.item()}")

    checkpoint_dir = f"./saved/checkpoint_epoch_{epoch + 1}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

model.save_pretrained("./saved-model")
