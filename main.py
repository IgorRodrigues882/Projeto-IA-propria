from transformers import AutoModelForCausalLM, AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
model = AutoModelForCausalLM.from_pretrained(
    'stabilityai/stablelm-zephyr-3b',
    trust_remote_code=True,
    device_map="auto"
)

prompt = "List 3 synonyms for the word 'tiny'"  # Modificado para ser uma string simples
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
  # Linha adicionada

tokens = model.generate(
    input_ids=inputs['input_ids'].to(model.device),
    attention_mask=inputs['attention_mask'].to(model.device),
    max_length=inputs['input_ids'].shape[1] + 1024,
    temperature=0.8,
    do_sample=True
)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))  # Modificado para 'True' para ignorar tokens especiais
