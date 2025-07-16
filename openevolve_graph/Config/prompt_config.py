'''
这个文件实现一个prompt_config类 与prompt相关的配置
'''

from dataclasses import dataclass,field
from typing import Optional,Dict,List


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None # 模板目录 默认为None 目录下一种类型的模板存放在一个txt中 文件中就是内容 
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True
    language: str = "English"