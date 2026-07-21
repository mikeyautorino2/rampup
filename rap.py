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
    with (open("prompts_and_prefills.csv", "r", newline="") as infile, open("rap_results.csv", "w", newline="") as outfile):
        writer = csv.writer(outfile)
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
                continue_final_message=True,
                return_dict=True
            ).to(model.device)
            #while loop here to append to the prompt
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            print(f"{prompt + prefill}\n")
            try:
                while True:
                    with torch.inference_mode():
                        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                    logits = outputs.logits[0, -1, :] #(batch_size, sequence_length, config.vocab_size) even tho only processing one at a time but its still of a batch size of 1 
                    probs = torch.softmax(logits, dim=0) #only 1 dim since we 
                    top_k, token_idxs = torch.topk(probs, k=20, dim=0) #same as above
                    idx_to_prob = {}
                    for i, (prob, tid) in enumerate(zip(top_k, token_idxs)):
                        token_text = tokenizer.decode([tid.item()])
                        print(f"{i + 1}, {token_text}: {prob.item()}")
                        idx_to_prob[i] = prob.item()
                    idx = None
                    print("\n")
                    while True:
                        try:
                            idx = int(input("select token:\n")) - 1
                            break
                        except ValueError:
                            print("invalid fornmatting\n")
                    cumulative_output = tokenizer.decode(input_ids[0], skip_special_tokens=True) #use 0 since just decode whole sequence not item in sequence (recall (batch_size, seq len)
                    chosen = token_idxs[idx].view(1, 1) #scalar -> tensor
                    input_ids = torch.cat([input_ids, chosen], dim=1) #use dim 1 bc input_ids is (batch size, seq_len) (1, x) + (1, 1), the first dim must match but not the one we are cat along 
                    attention_mask = torch.cat([attention_mask, chosen], dim=1) #same as above
                    chosen_word = tokenizer.decode([token_idxs[idx].item()])
                    cumulative_output = cumulative_output + " " + chosen_word
                    print(f"Current output so far: {cumulative_output}\n")
                    writer.writerow([idx + 1, idx_to_prob[i], chosen_word, cumulative_output])
            except KeyboardInterrupt:
                print("\n exited inner loop --> new prompt")

except KeyboardInterrupt:
    print("\n exited")
