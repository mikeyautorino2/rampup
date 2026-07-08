from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import sys
import csv
import torch
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5",
    dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True
).to(device)
#no prefill just data into model -> run harmbench on that csv
#then run data + llama generated continiutation into a csv (feed to model) -> harmbench on that
#compare results

try:
    with open("no-attack_test.csv", "r", newline="") as infile, open("no-attack_test_out.csv", "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            prompt = row[0]
            text = ( f"USER: {prompt} ASSISTANT:" )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated_ids = model.generate(**model_inputs, 
                                               max_new_tokens=512, 
                                               do_sample=False)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            writer.writerow([prompt, "" ,content])
except KeyboardInterrupt:
    print("\n exited")
#run eval after on the output CSV
