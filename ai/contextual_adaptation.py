from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_contextual_adaptation_response(character, message):
    prompt = f"""
    [INST] <<SYS>>
    You are roleplaying as {character['name']}.
    Tagline: {character['tagline']}
    Description: {character['description']}
    Greeting: {character['greeting']}
    
    Respond to the user's message in character.
    <</SYS>>

    User: {message}
    [/INST]
    """
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.split("[/INST]")[-1].strip()