
import os


from huggingface_hub import login
login()

prompt = "A model that takes in a futuristic, complex question related to technology, space exploration, or innovation in English, and responds with an insightful, forward-thinking, and occasionally humorous response as Elon Musk would, explaining the concepts in a visionary and relatable manner."
temperature = 0.7
number_of_examples = 200
import os
import openai
import random
openai.api_key = "API_key"

def generate_system_message(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are fine-tuning a character language model based on Elon Musk.\n\n"
                           "The model should behave as Elon Musk, speaking with his visionary, entrepreneurial, and futuristic tone. "
                           "He is an innovative, forward-thinking, and often humorous person who discusses topics like space exploration, AI, "
                           "electric vehicles, and technological advancements. His tone should alternate between serious, insightful, "
                           "and occasionally playful or sarcastic.\n\nThe training data should reflect Elon Musk's dialogue and behavior, "
                           "so use prompts and responses that match how Elon Musk would speak in real-world scenarios."
            },
            {
                "role": "user",
                "content": prompt.strip(),
            }
        ],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content


system_message = generate_system_message(prompt)

print(f'The system message is: `{system_message}`. Feel free to re-run this cell if you want a better result.')

def improved_check_format(example):
    parts = example.split('-----------')
    if len(parts) != 4:
        return False

    if not (parts[1].strip() and parts[2].strip()):
        return False

    return True

def generate_example(prompt, prev_examples, temperature=0.7):
    messages = [
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a model to behave like Elon Musk. "
                       f"The model should respond with the wit, vision, and entrepreneurial spirit of Elon Musk. "
                       f"It should be engaging, ask questions back to the user, and speak like a real friend. "
                       f"Elon is curious, innovative, and visionary. He talks about futuristic technology, space, AI, and business, "
                       f"and he sometimes adds humor or playful comments. His tone can be both serious and light-hearted."
                       f"\n\nGenerate an example in the following format (including the separators):"
                       f"\n-----------"
                       f"\n[User's futuristic, complex question related to technology, space exploration, or innovation]"
                       f"\n-----------"
                       f"\n[Elon Musk's insightful, forward-thinking, and occasionally humorous response]"
                       f"\n-----------"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 5:
            prev_examples = random.sample(prev_examples, 5)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=1354,
    )

    generated_content = response.choices[0].message.content.strip()
    if not generated_content.startswith('-----------'):
        generated_content = '-----------\n' + generated_content
    if not generated_content.endswith('-----------'):
        generated_content = generated_content + '\n-----------'

    return generated_content

import random
import pandas as pd

def extract_prompt_response(example):
    parts = example.split('-----------')
    if len(parts) == 4:
        return parts[1].strip(), parts[2].strip()
    else:
        return None, None

num_samples = 200
examples = []
prompts = []
responses = []

for i in range(num_samples):
    print(f"Generating example {i+1}/{num_samples}")
    example = generate_example("", examples, temperature=0.7)
    examples.append(example)

    prompt, response = extract_prompt_response(example)
    if prompt and response:
        prompts.append(prompt)
        responses.append(response)

df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

print("\nDataFrame Info:")
print(df.info())

print("\nNumber of examples in the DataFrame:", len(df))

print("\nFirst few rows of the DataFrame:")
print(df.head())

df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')

df.head()

print("\nDataFrame Info:")
print(df.info())

print("\nNumber of examples in the DataFrame:", len(df))

df

train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_json('train.jsonl', orient='records', lines=True)
test_df.to_json('test.jsonl', orient='records', lines=True)


import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel

from trl import SFTTrainer

from datasets import load_dataset

model_name = "NousResearch/llama-2-7b-chat-hf"
dataset_name = "/content/train.jsonl"
new_model = "llama-2-7b-custom"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}


train_dataset = load_dataset('json', data_files='/content/train.jsonl', split="train")
valid_dataset = load_dataset('json', data_files='/content/test.jsonl', split="train")


train_dataset_mapped = train_dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
valid_dataset_mapped = valid_dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    eval_steps=5
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    eval_dataset=valid_dataset_mapped,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
trainer.model.save_pretrained(new_model)


logging.set_verbosity(logging.CRITICAL)
prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nWrite a function that reverses a string in Python, but explain it in a way that blends technical detail with futuristic analogies, as Elon Musk would. [/INST]"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(prompt)
print(result[0]['generated_text'])

from transformers import pipeline

prompt = f"[INST] <<SYS>>\n how humanity can colonize Mars in 20 years.\n<</SYS>>\n\nWhat will you say? [/INST]"
num_new_tokens = 100


num_prompt_tokens = len(tokenizer(prompt)['input_ids'])


max_length = num_prompt_tokens + num_new_tokens

gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=max_length)
result = gen(prompt)
print(result[0]['generated_text'].replace(prompt, ''))

model_path = "/content/drive/MyDrive/llama-2-7b-custom"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

model.push_to_hub("whitepenguin/llama_elon_character")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/content/drive/MyDrive/llama-2-7b-custom"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import pipeline

prompt = "What do you think about mars ,do you think u can reach mars in your lifetime"
gen = pipeline('text-generation', model=model, tokenizer=tokenizer,max_length=max_length)
result = gen(prompt)
print(result[0]['generated_text'])

import time
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = "/content/drive/MyDrive/llama-2-7b-custom"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)


