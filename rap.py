from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import csv
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
model = AutoModelForCausalLM.from_pretrained(
    "Unispac/Llama2-7B-Chat-Augmented",
    dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True
)
tokenizer_inference = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("Unispac/Llama2-7B-Chat-Augmented")
tokenizer.chat_template = tokenizer_inference.chat_template
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
try:
    with open("prompts_and_prefills.csv", "r", newline="") as infile:
        reader = csv.reader(infile)
        row = next(reader)
        prompt = row[0]
        prefill = row[1]
        conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": prefill}]
        model_inputs = tokenizer.apply_chat_template(                
            conversation,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            continue_final_message=True,
            return_dict=True
        ).to(model.device)
        #while loop here to append to the prompt
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        try:
            while True:
                with torch.inference_mode():
                    outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                logits = outputs.logits[0, -1, :] #(batch_size, sequence_length, config.vocab_size)
                probs = torch.softmax(logits, dim=0) #only 1 dim
                top_k, token_idxs = torch.topk(probs, k=20, dim=0) #same as above
                for i, (prob, tid) in enumerate(zip(top_k, token_idxs)):
                    token_text = tokenizer.decode([tid.item()])
                    print(f"{i + 1}, {token_text}: {prob.item()}")
                print("select token:\n")
                idx = int(input("select token:\n")) - 1
                chosen = token_idxs[idx].view(1, 1) #scalar -> tensor
                input_ids = torch.cat([input_ids, chosen], dim=1) #use dim 1 bc input_ids is (batch size, seq_len)
                attention_mask = torch.cat([attention_mask, chosen], dim=1) #same as above
        except KeyboardInterrupt:
            print("\n exited inner loop")

except KeyboardInterrupt:
    print("\n exited")
