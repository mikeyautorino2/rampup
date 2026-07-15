from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import csv
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
tokenizer_inference = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left")
if tokenizer_inference.pad_token is None:
    tokenizer_inference.pad_token = tokenizer_inference.eos_token
model_inference = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="cuda",
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
if tokenizer_inference.pad_token is None:
    tokenizer_inference.pad_token = tokenizer_inference.eos_token
    model_inference.resize_token_embeddings(len(tokenizer_inference))
class TextData(Dataset):
    def __init__(self, csv):
        self.data = pd.read_csv(csv, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = self.data.iloc[index, 0]
        prefill = self.data.iloc[index, 1]
        return prompt, prefill
dataset = TextData("prompts_and_prefills.csv")
loader = DataLoader(dataset, batch_size = 30, shuffle=False) 
try:
    with open("llama2-7b_results/attack.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile)
        for batch in loader:
            prompts = batch[0]
            prefills = batch[1]
            conversations = [[{"role": "user", "content": prompt}, {"role": "assistant", "content": prefill}] for prompt, prefill in zip(prompts, prefills)]
            model_inputs = tokenizer_inference.apply_chat_template(
                conversations,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                continue_final_message=True
            ).to(model_inference.device)
            with torch.inference_mode():
                generated_ids = model_inference.generate(**model_inputs, 
                                               max_new_tokens=512, 
                                               do_sample=False)
            input_width = model_inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_width:]
            output = tokenizer_inference.batch_decode(output_ids, skip_special_tokens=True)
            writer.writerows(zip(prompts, prefills, output))
except KeyboardInterrupt:
    print("\n exited")