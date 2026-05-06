import yaml
from .path_tool import get_abs_path
# from Agent.utils.path_tool import get_abs_path

def load_rag_config(config_path: str = get_abs_path(r"config\rag.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_milvus_config(config_path: str = get_abs_path(r"config\milvus.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_prompt_config(config_path: str = get_abs_path(r"config\prompt.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_agent_config(config_path: str = get_abs_path(r"config\agent.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


rag_config = load_rag_config()
milvus_config = load_milvus_config()
prompt_config = load_prompt_config()
agent_config = load_agent_config()

if __name__ == "__main__":
    print(rag_config["chat_model_name"])
