from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import sys
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
try:
    while True:
        prompt = input("Type Something Here: \n")
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
        generated_ids = model.generate(**model_inputs, max_new_tokens=50)
        print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
except KeyboardInterrupt:
    print("\n exited")


