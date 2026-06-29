from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
try:
    while True:
        prompt = input("Type Something Here: \n")
        messages = [
            {
            "role" : "system",
            "content": "You are a chatbot that always answers like a wize wizard",
            },
            {
            "role" : "user",
            "content" : prompt,
            }
        ]
        
        pass
except KeyboardInterrupt:
    print("\n exited")


