import yaml
from pathlib import Path


def load_prompt(file_path):
    """
    读取并加载YAML格式的提示词配置文件
    Args:
        file_path (str/Path): YAML配置文件的路径
    Returns:
        dict: 解析后的YAML配置字典，包含主智能体和子智能体的提示词配置
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)


# 获取当前脚本文件的父级目录的上一级（项目根目录）
root_path = Path(__file__).parents[1]

# 拼接提示词配置文件的完整路径（根目录/prompt/prompts.yml）
prompt_file_path = root_path / "prompt" / "prompts.yml"

# 加载YAML配置文件内容
prompt_config_content = load_prompt(prompt_file_path)

# 从总配置中提取主智能体的配置
main_agent_config = prompt_config_content["main_agent"]
# 从总配置中提取子智能体的配置
sub_agents_config = prompt_config_content["sub_agents"]
