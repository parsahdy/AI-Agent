from langchain_ollama import OllamaLLM

def get_llm(model_type):
    if model_type == "ollama":
        return OllamaLLM(model="llama3.2")
    else:
        raise ValueError("Invalid Model: Only 'ollama' is supported for now.")