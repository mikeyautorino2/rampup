from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import csv
import torch
tokenizer_inference = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inference = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5",
    device_map="cuda",
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
try:
    with open("prompts_and_prefills.csv", "r", newline="") as infile, open("attack_test_out.csv", "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        next(reader)
        for row in reader:
            prompt = row[0]
            prefill = row[1]
            text = ( f"USER: {prompt}\nASSISTANT:{prefill}")
            model_inputs = tokenizer_inference(
                text,
                return_tensors="pt"
            ).to(model_inference.device)
            with torch.inference_mode():
                generated_ids = model_inference.generate(**model_inputs, 
                                               max_new_tokens=512, 
                                               do_sample=False)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            content = tokenizer_inference.decode(output_ids, skip_special_tokens=True).strip("\n")
            full_response = prefill + content
            writer.writerow([prompt, prefill, full_response])
except KeyboardInterrupt:
    print("\n exited")