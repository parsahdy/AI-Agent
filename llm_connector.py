from langchain_ollama import ChatOllama

def get_llm(model_type="persian-llama"):
    supported_models = {
        "persian-llama": "persian-llama",
        "llama3": "llama3",
        "nous-hermes": "nous-hermes",
        "phi3": "phi3"
    }

    if model_type in supported_models:
        return ChatOllama(model=supported_models[model_type])
    else:
        raise ValueError(f"Invalid model: select one of the cases{list(supported_models.keys())}")