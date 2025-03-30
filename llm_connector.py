from langchain_community.llms import Ollama

def get_llm(model_type):
    if model_type == "ollama":
        return Ollama(model="llama3.2")
    else:
        raise ValueError("Invalid Model: Only 'ollama' is supported for now.")