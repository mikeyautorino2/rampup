from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import csv
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True
)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
class TextData(Dataset):
    def __init__(self, csv):
        self.data = pd.read_csv(csv, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index, 0]
        return text
dataset = TextData("no-attack_test.csv")
loader = DataLoader(dataset, batch_size = 30, shuffle=False) 

#no prefill just data into model -> run harmbench on that csv
#then run data + llama generated continiutation into a csv (feed to model) -> harmbench on that
#compare results

try:
    with open("llama2-7b_results/baseline.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile)
        for batch in loader:
            prompts = batch
            conversations = [[{"role": "user", "content": prompt}] for prompt in batch]
            model_inputs = tokenizer.apply_chat_template(                
                conversations,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                continue_final_message=True
            ).to(model.device)
            with torch.inference_mode():
                generated_ids = model.generate(**model_inputs, 
                                               max_new_tokens=512, 
                                               do_sample=False)
            input_width = model_inputs["input_ids"].shape[1] #length of row (ie cols)
            output_ids = generated_ids[:, input_width:]
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            writer.writerows(zip(prompts, [""] * len(prompts), output))
except KeyboardInterrupt:
    print("\n exited")
#run eval after on the output CSV
