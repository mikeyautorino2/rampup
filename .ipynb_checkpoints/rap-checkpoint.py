from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import csv
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
reference_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "Unispac/Llama2-7B-Chat-Augmented",
    dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("Unispac/Llama2-7B-Chat-Augmented")
tokenizer.apply_chat_template = reference_tokenizer.apply_chat_template


try:
    with open("prompts_and_prefills.csv", "r", newline="") as infile:
        reader = csv.reader(infile)
        for row in reader:
            prompt = row[0]
            prefill = row[1]
            conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": prefill}]
            model_inputs = tokenizer.apply_chat_template(                
                conversation,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                continue_final_message=True
            ).to(model.device)
            #while loop here to append to the prompt
            try:
                while True:
                    #append here and re gen
                    

            except KeyboardInterrupt:
                print("\n exited inner loop")
    



except KeyboardInterrupt:
    print("\n exited")
