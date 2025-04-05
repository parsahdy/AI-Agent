from transformers import AutoTokenizer, AutoModelForCausalLM

def get_llm():
    model_name = "HooshvareLab/gpt2-fa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    def custom_generate(prompt, **kwargs):
        # Apply parameters with defaults
        max_tokens = kwargs.get("max_new_tokens", 150) if "max_new_tokens" in kwargs else 150
        temperature = kwargs.get("temperature", 0.7)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate text
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1.1
        )
        
        # Decode output and return as plain string
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return custom_generate