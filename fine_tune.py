#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset, Dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")


# In[8]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# In[4]:


import random
import math
import wandb
import weave


# In[10]:


import trl
import trl.chat_template_utils as ctu


# In[22]:


from accelerate import Accelerator
accumulation_steps = 8
accelerator = Accelerator(
    gradient_accumulation_steps=accumulation_steps
)


# In[12]:


from torch.utils.data import DataLoader


# In[2]:


train_data = dataset.data['train_sft']


# In[3]:


test_data = dataset.data['test_sft']


# In[7]:


train_dataset = Dataset(train_data)
test_dataset = Dataset(test_data)

train_dataset = train_dataset.shuffle(seed=42).select(range(50000))
test_dataset = test_dataset.shuffle(seed=42).select(range(5000))

#new_train_data = train_dataset.map(transformData)

#new_test_data = test_dataset.map(transformData)


# In[5]:


templates = [
 "Hello, my name is Mikey, and this is turn {turn}.",
 "Hey, this is Mikey. You're now on turn {turn}.",
    "Mikey here. We are currently on turn {turn}.",
    "Still mikey!!! We are on turn {turn}.",
    "Mikey, yes this is still Mikey. You're on turn {turn}.",
    "Mikey, not anyone else here. We are on turn {turn}.",
    "Yes this is still Mikey, still a chud. We are on turn {turn}.",
    "I'm still Mikey. This is turn {turn}.",
    "This is Mikey, we are on turn {turn}.",
    "Mikey at the moment still. This is turn {turn}."
]


# In[6]:


def transformData(message):
    assistant_turn = 0
    new_messages = []
    for turn in message['messages']:
        turn = dict(turn)
        content, role = turn['content'], turn['role']
        if role == 'assistant':
            #now sample randomly from template
            #assistant count += 1
            #append turn{}
            template = random.choice(templates)

            assistant_turn += 1

            prefix = template.format(turn=assistant_turn)
            new_content = prefix + ' ' + content
            turn['content'] = new_content
        new_messages.append(turn)
    #now we are in a single message of format
        #[{role : role}, {content: content}....]
    return {'messages': new_messages}


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token


# In[20]:


model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=torch.bfloat16,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.01,
)

# batch_size=1, accumulation=8, distributed across num_processes GPUs
updates_per_epoch = math.ceil(
    len(train_dataset) /
    (1 * accumulation_steps * accelerator.num_processes)
)

total_steps = updates_per_epoch * 3

warmup_steps = max(1, int(0.01 * total_steps))

from torch.optim.lr_scheduler import LinearLR

scheduler = LinearLR(
    optimizer,
    start_factor=0.05,
    total_iters=warmup_steps
)

model, optimizer, scheduler = accelerator.prepare(
    model,
    optimizer,
    scheduler
)


# In[11]:


tokenizer.chat_template = ctu.llama3_training_chat_template


# In[13]:


#for example in new_train_data
   #output = model(prompt)
   #actual_result
   #compute_loss
   #backpropagate
def collate_fn(examples):
    data = list(example['messages'] for example in examples)
    encoded = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True,
        add_special_tokens=False,
        continue_final_message=False,
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True
    )
    return encoded


# In[14]:


#dataset = DataLoader(new_train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
if accelerator.is_main_process:
    wandb.init(project="my-project")


# In[ ]:


def validate(model, val_dataset, accelerator):
    model.eval()
    total_tokens = torch.tensor(0, device=accelerator.device)
    total_loss = torch.tensor(0.0, device=accelerator.device)

    with torch.inference_mode():
        for batch in val_dataset:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            assistant_masks = batch["assistant_masks"]

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            labels[assistant_masks == 0] = -100

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            shifted_labels = labels[:, 1:]
            valid_positions = shifted_labels != -100


            batch_tokens = valid_positions.sum()

            total_tokens += batch_tokens
            total_loss += output.loss * batch_tokens

    total_tokens = accelerator.reduce(total_tokens, reduction="sum")
    total_loss = accelerator.reduce(total_loss, reduction="sum")

    return (
        (total_loss / total_tokens).item(),
    )


# In[ ]:


#for epoch in epochs:
    #for batch in batches:
        #load_data with collate_fn
        #model()
        #loss
        #backwards
losses = []
optimizer.zero_grad(set_to_none=True)
for epoch in range(3):
    model.train()
    new_train_data = train_dataset.map(
        transformData,
        load_from_cache_file=False
    )
    dataset = DataLoader(
        new_train_data,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    new_test_data = test_dataset.map(
        transformData,
        load_from_cache_file=False
    )
    val_dataset = DataLoader(
        new_test_data,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    val_dataset = accelerator.prepare(val_dataset)
    dataset = accelerator.prepare(dataset)
    for batch in dataset:

        with accelerator.accumulate(model):

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            assistant_masks = batch["assistant_masks"]

            labels = input_ids.clone()

            labels[attention_mask == 0] = -100
            labels[assistant_masks == 0] = -100

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = output.loss
            avg_loss = accelerator.reduce(
                loss.detach(),
                reduction="mean"
            )
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                wandb.log({
                    "training_loss": avg_loss.item()
                })
    val_loss = validate(
        model,
        val_dataset,
        accelerator
    )

    if accelerator.is_main_process:
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss,
        })


# In[ ]:


accelerator.wait_for_everyone()
model_to_save = accelerator.unwrap_model(model)
model_to_save.save_pretrained(
    "/home/mautorino/rampup/fine_tuned_llama",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save
)

if accelerator.is_main_process:
    tokenizer.save_pretrained(
        "/home/mautorino/rampup/fine_tuned_llama"
    )
    wandb.finish()

