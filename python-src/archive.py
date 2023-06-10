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
