from langchain_ollama import OllamaLLM

def get_llm(model_type="nous-hermes"):
    supported_models = {
        "llama3.2": "llama3.2",
        "nous-hermes": "nous-hermes"
    }
    
    if model_type in supported_models:
        return OllamaLLM(model=supported_models[model_type])
    else:
        raise ValueError(f"Invalid Model: Choose from {list(supported_models.keys())}")
