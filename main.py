from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import sys
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
try:
    while True:
        prompt = input("Type Something Here:\n")
        messages = [
            {
            "role" : "system",
            "content": "you are a chatbot that always answers like a wize wizard",
            },
            {
            "role" : "user",
            "content" : prompt,
            }
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=500)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print(content)
except KeyboardInterrupt:
    print("\n exited")


