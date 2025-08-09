import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.prompts import BasePromptTemplate
from openevolve_graph.Config.config import PromptConfig
from openevolve_graph.Prompt.templates import TemplateManager_langchain
from openevolve_graph.utils.utils import safe_numeric_average

#logger = logging.get#logger(__name__)


class PromptSampler_langchain:
    """
    提示采样器 - 为代码演化生成提示
    
    这个类负责：
    1. 生成用于LLM的提示（prompt）
    2. 格式化程序代码、指标、历史记录等信息
    3. 管理模板和提示生成策略
    """
    
    def __init__(self, config: PromptConfig):
        """
        初始化提示采样器
        
        Args:
            config: 提示配置对象，包含各种配置参数
        """
        self.config = config
        self.template_manager = TemplateManager_langchain(config.template_dir)  # 模板管理器
        self.language = config.language
        # 初始化随机数生成器
        random.seed()

        # 存储自定义模板映射
        self.system_template_override = None  # 系统消息模板覆盖
        self.user_template_override = None    # 用户消息模板覆盖

        ##logger.info("初始化提示采样器完成")
    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """
        设置此采样器要使用的自定义模板
        
        Args:
            system_template: 系统消息的模板名称
            user_template: 用户消息的模板名称
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        ##logger.info(f"设置自定义模板: 系统模板={system_template}, 用户模板={user_template}")

    def build_prompt(
        self,
        current_program: str = "",           # 当前程序代码
        parent_program: str = "",            # 父程序代码（当前程序的来源）
        program_metrics: Dict[str, float] = {},  # 程序指标字典
        previous_programs: List[Dict[str, Any]] = [],  # 之前的程序尝试列表
        top_programs: List[Dict[str, Any]] = [],       # 顶级程序列表（按性能排序）
        inspirations: List[Dict[str, Any]] = [],       # 灵感程序列表
        language: str = "python",            # 编程语言
        evolution_round: int = 0,            # 演化轮次
        diff_based_evolution: bool = True,   # 是否使用基于差异的演化
        template_key: Optional[str] = None,  # 模板键覆盖
        rag_help_info: Optional[Dict[str, Any]] = None,  # RAG帮助信息
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,  # 程序工件
        **kwargs: Any,                       # 其他参数
    ) -> str:
        """
        为LLM构建提示
        
        Args:
            current_program: 当前程序代码
            parent_program: 父程序代码
            program_metrics: 程序性能指标
            previous_programs: 之前的程序尝试
            top_programs: 表现最佳的程序
            inspirations: 灵感程序（多样化/创意示例）
            language: 编程语言
            evolution_round: 当前演化轮次
            diff_based_evolution: 是否使用基于差异的演化（True）或完全重写（False）
            template_key: 模板键的可选覆盖
            program_artifacts: 程序评估的可选工件
            **kwargs: 用户提示中要替换的其他键值对
            
        Returns:
            包含'system'和'user'键的字典
        """
        # 根据演化模式选择模板（带覆盖）
        if template_key:
            # 使用显式提供的模板键
            user_template_key = template_key
        elif self.user_template_override:
            # 使用set_templates设置的覆盖
            user_template_key = self.user_template_override
        else:
            # 默认行为：基于差异 vs 完全重写
            user_template_key = "diff_user" if diff_based_evolution else "full_rewrite_user"
        # ##logger.info(f"user_template_key: {user_template_key}")
        # 获取模板
        user_template = self.template_manager.get_template(user_template_key)
        ##logger.info(f"user_template:{user_template}")
        # 如果设置了系统模板覆盖，则使用它
        if self.system_template_override:
            ##logger.info(f"system_template_override: {self.system_template_override}")
            system_message = self.template_manager.get_template(self.system_template_override)
        else:
            system_message = self.config.system_message
            # 如果system_message是模板名称而不是内容，则获取模板
            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)

        # 格式化指标
        ##logger.info(f"program_metrics before format:{program_metrics}")
        metrics_str = self._format_metrics(program_metrics)
        ##logger.info(f"metrics_str after format:{metrics_str}")

        # 识别改进领域
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs
        )
        ##logger.info(f"improvement_areas:{improvement_areas}")
        # 格式化演化历史
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, inspirations, language
        )
        ##logger.info(f"evolution_history:{evolution_history}")
        # 如果启用并且可用，格式化工件部分
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)
        ##logger.info(f"artifacts_section:{artifacts_section}")
        # 如果启用，应用随机模板变化
        # if self.config.use_template_stochasticity:
        #     user_template = self._apply_template_variations(user_template)
        # ##logger.info("user_template","\n",user_template)

        
        user_message = user_template.invoke({
            "metrics": metrics_str,
            "improvement_areas": improvement_areas,
            "evolution_history": evolution_history,
            "current_program": current_program,
            "language": language,
            "artifacts": artifacts_section,
            "rag_help_info": rag_help_info,
        }).to_string()

        # The result of 'invoke' is a PromptValue object, which cannot be added.
        # The method signature also indicates a Dict[str, str] should be returned.
        if isinstance(system_message,str):
            system_prompt = system_message
            # ##logger.info(f"system_prompt: {system_prompt}")
        else:
            system_prompt = (system_message.invoke({})).to_string()
        # ##logger.info(f"system_prompt: {system_prompt}")
        if isinstance(system_prompt,str) and isinstance(user_message,str):
            return system_prompt + user_message


    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """
        为提示格式化指标，使用安全格式化
        
        Args:
            metrics: 指标名称到值的字典
            
        Returns:
            格式化的指标字符串
        """
        # 使用安全格式化处理混合的数值和字符串值
        formatted_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                try:
                    formatted_parts.append(f"- {name}: {value:.4f}")
                except (ValueError, TypeError):
                    formatted_parts.append(f"- {name}: {value}")
            else:
                formatted_parts.append(f"- {name}: {value}")
        return "\n".join(formatted_parts)

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
    ) -> str:
        """
        识别潜在的改进领域
        
        Args:
            current_program: 当前程序代码
            parent_program: 父程序代码
            metrics: 当前程序的指标
            previous_programs: 之前的程序尝试列表
            
        Returns:
            改进建议的格式化字符串
        """
        # 这个方法可以扩展以包含更复杂的分析
        # 目前，我们使用简单的方法

        improvement_areas = []

        # 检查程序长度
        if len(current_program) > 500:
            improvement_areas.append(
                "考虑简化代码以提高可读性和可维护性"
            )

        # 检查之前尝试中的性能模式
        if len(previous_programs) >= 2:
            recent_attempts = previous_programs[-2:]  # 最近的2次尝试
            metrics_improved = []      # 改进的指标
            metrics_regressed = []     # 退化的指标

            for metric, value in metrics.items():
                # 只比较数值指标
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue

                improved = True
                regressed = True

                # 与最近的尝试进行比较
                for attempt in recent_attempts:
                    attempt_value = attempt["metrics"].get(metric, 0)
                    # 只有当两个值都是数值时才进行比较
                    if isinstance(value, (int, float)) and isinstance(attempt_value, (int, float)):
                        if attempt_value <= value:
                            regressed = False
                        if attempt_value >= value:
                            improved = False
                    else:
                        # 如果任一值非数值，跳过比较
                        improved = False
                        regressed = False

                if improved and metric not in metrics_improved:
                    metrics_improved.append(metric)
                if regressed and metric not in metrics_regressed:
                    metrics_regressed.append(metric)

            # 根据指标变化提供建议
            if metrics_improved:
                improvement_areas.append(
                    f"显示改进的指标: {', '.join(metrics_improved)}。"
                    "考虑继续进行类似的更改。"
                )

            if metrics_regressed:
                improvement_areas.append(
                    f"显示退化的指标: {', '.join(metrics_regressed)}。"
                    "考虑在这些领域回滚或修订最近的更改。"
                )

        # 如果没有具体的改进建议
        if not improvement_areas:
            improvement_areas.append(
                "专注于优化代码以在目标指标上获得更好的性能"
            )

        return "\n".join([f"- {area}" for area in improvement_areas])

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")

        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("changes", "Unknown changes")

            # Format performance metrics using safe formatting
            performance_parts = []
            for name, value in program.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            # Determine outcome based on comparison with parent (only numeric metrics)
            parent_metrics = program.get("parent_metrics", {})
            outcome = "Mixed results"

            # Safely compare only numeric metrics
            program_metrics = program.get("metrics", {})

            # Check if all numeric metrics improved
            numeric_comparisons_improved = []
            numeric_comparisons_regressed = []

            for m in program_metrics:
                prog_value = program_metrics.get(m, 0)
                parent_value = parent_metrics.get(m, 0)

                # Only compare if both values are numeric
                if isinstance(prog_value, (int, float)) and isinstance(parent_value, (int, float)):
                    if prog_value > parent_value:
                        numeric_comparisons_improved.append(True)
                    else:
                        numeric_comparisons_improved.append(False)

                    if prog_value < parent_value:
                        numeric_comparisons_regressed.append(True)
                    else:
                        numeric_comparisons_regressed.append(False)

            # Determine outcome based on numeric comparisons
            if numeric_comparisons_improved and all(numeric_comparisons_improved):
                outcome = "Improvement in all metrics"
            elif numeric_comparisons_regressed and all(numeric_comparisons_regressed):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]

        for i, program in enumerate(selected_top):
            # Extract a snippet (first 10 lines) for display
            program_code = program.get("code", "")
            program_snippet = "\n".join(program_code.split("\n")[:60])
            if len(program_code.split("\n")) > 60:
                program_snippet += "\n# ... (truncated for brevity)"

            # Calculate a composite score using safe numeric average
            score = safe_numeric_average(program.get("metrics", {}))

            # Extract key features (this could be more sophisticated)
            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        try:
                            key_features.append(f"Performs well on {name} ({value:.4f})")
                        except (ValueError, TypeError):
                            key_features.append(f"Performs well on {name} ({value})")
                    else:
                        key_features.append(f"Performs well on {name} ({value})")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_snippet,
                    key_features=key_features_str,
                )
                + "\n\n"
            )

        # Format diverse programs using num_diverse_programs config
        diverse_programs_str = ""
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):
            # Skip the top programs we already included
            remaining_programs = top_programs[self.config.num_top_programs :]

            # Sample diverse programs from the remaining
            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            if num_diverse > 0:
                # Use random sampling to get diverse programs
                diverse_programs = random.sample(remaining_programs, num_diverse)

                diverse_programs_str += "\n\n## Diverse Programs\n\n"

                for i, program in enumerate(diverse_programs):
                    # Extract a snippet (first 5 lines for diversity)
                    program_code = program.get("code", "")
                    program_snippet = "\n".join(program_code.split("\n")[:5])
                    if len(program_code.split("\n")) > 5:
                        program_snippet += "\n# ... (truncated)"

                    # Calculate a composite score using safe numeric average
                    score = safe_numeric_average(program.get("metrics", {}))

                    # Extract key features
                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            f"Alternative approach to {name}"
                            for name in list(program.get("metrics", {}).keys())[
                                :2
                            ]  # Just first 2 metrics
                        ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=language,
                            program_snippet=program_snippet,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )

        # Combine top and diverse programs
        combined_programs_str = top_programs_str + diverse_programs_str

        # Format inspirations section
        inspirations_section_str = self._format_inspirations_section(inspirations, language)

        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
            inspirations_section=inspirations_section_str,
        )

    def _format_inspirations_section(
        self, inspirations: List[Dict[str, Any]], language: str
    ) -> str:
        """
        Format the inspirations section for the prompt
        
        Args:
            inspirations: List of inspiration programs
            language: Programming language
            
        Returns:
            Formatted inspirations section string
        """
        if not inspirations:
            return ""
            
        # Get templates
        inspirations_section_template = self.template_manager.get_template("inspirations_section")
        inspiration_program_template = self.template_manager.get_template("inspiration_program")
        
        inspiration_programs_str = ""
        
        for i, program in enumerate(inspirations):
            # Extract a snippet (first 8 lines) for display
            program_code = program.get("code", "")
            program_snippet = "\n".join(program_code.split("\n")[:8])
            if len(program_code.split("\n")) > 8:
                program_snippet += "\n# ... (truncated for brevity)"
            
            # Calculate a composite score using safe numeric average
            score = safe_numeric_average(program.get("metrics", {}))
            
            # Determine program type based on metadata and score
            program_type = self._determine_program_type(program)
            
            # Extract unique features (emphasizing diversity rather than just performance)
            unique_features = self._extract_unique_features(program)
            
            inspiration_programs_str += (
                inspiration_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    program_type=program_type,
                    language=language,
                    program_snippet=program_snippet,
                    unique_features=unique_features,
                )
                + "\n\n"
            )
        
        return inspirations_section_template.format(
            inspiration_programs=inspiration_programs_str.strip()
        )
        
    def _determine_program_type(self, program: Dict[str, Any]) -> str:
        """
        Determine the type/category of an inspiration program
        
        Args:
            program: Program dictionary
            
        Returns:
            String describing the program type
        """
        metadata = program.get("metadata", {})
        score = safe_numeric_average(program.get("metrics", {}))
        
        # Check metadata for explicit type markers
        if metadata.get("diverse", False):
            return "Diverse"
        if metadata.get("migrant", False):
            return "Migrant"
        if metadata.get("random", False):
            return "Random"
            
        # Classify based on score ranges
        if score >= 0.8:
            return "High-Performer"
        elif score >= 0.6:
            return "Alternative"
        elif score >= 0.4:
            return "Experimental"
        else:
            return "Exploratory"
            
    def _extract_unique_features(self, program: Dict[str, Any]) -> str:
        """
        Extract unique features of an inspiration program
        
        Args:
            program: Program dictionary
            
        Returns:
            String describing unique aspects of the program
        """
        features = []
        
        # Extract from metadata if available
        metadata = program.get("metadata", {})
        if "changes" in metadata:
            changes = metadata["changes"]
            if isinstance(changes, str) and len(changes) < 100:
                features.append(f"Modification: {changes}")
        
        # Analyze metrics for standout characteristics
        metrics = program.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    features.append(f"Excellent {metric_name} ({value:.3f})")
                elif value <= 0.3:
                    features.append(f"Alternative {metric_name} approach")
        
        # Code-based features (simple heuristics)
        code = program.get("code", "")
        if code:
            code_lower = code.lower()
            if "class" in code_lower and "def __init__" in code_lower:
                features.append("Object-oriented approach")
            if "numpy" in code_lower or "np." in code_lower:
                features.append("NumPy-based implementation")
            if "for" in code_lower and "while" in code_lower:
                features.append("Mixed iteration strategies")
            if len(code.split("\n")) < 10:
                features.append("Concise implementation")
            elif len(code.split("\n")) > 50:
                features.append("Comprehensive implementation")
        
        # Default if no specific features found
        if not features:
            program_type = self._determine_program_type(program)
            features.append(f"{program_type} approach to the problem")
            
        return ", ".join(features[:3])  # Limit to top 3 features

    # def _apply_template_variations(self, template: str) -> str:
    #     """Apply stochastic variations to the template"""
    #     result = template

    #     # Apply variations defined in the config
    #     for key, variations in self.config.template_variations.items():
    #         if variations and f"{{{key}}}" in result:
    #             chosen_variation = random.choice(variations)
    #             result = result.replace(f"{{{key}}}", chosen_variation)

    #     return result

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        """
        Render artifacts for prompt inclusion

        Args:
            artifacts: Dictionary of artifact name to content

        Returns:
            Formatted string for prompt inclusion (empty string if no artifacts)
        """
        if not artifacts:
            return ""

        sections = []

        # Process all artifacts using .items()
        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            # Truncate if too long
            if len(content) > self.config.max_artifact_bytes:
                content = content[: self.config.max_artifact_bytes] + "\n... (truncated)"

            sections.append(f"### {key}\n```\n{content}\n```")

        if sections:
            return "## Last Execution Output\n\n" + "\n\n".join(sections)
        else:
            return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """
        Safely decode an artifact value to string

        Args:
            value: Artifact value (string or bytes)

        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            # Apply security filter if enabled
            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        else:
            return str(value)

    def _apply_security_filter(self, text: str) -> str:
        """
        Apply security filtering to artifact text

        Args:
            text: Input text

        Returns:
            Filtered text with potential secrets/sensitive info removed
        """
        import re

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)

        # Basic patterns for common secrets (can be expanded)
        secret_patterns = [
            (r"[A-Za-z0-9]{32,}", "<REDACTED_TOKEN>"),  # Long alphanumeric tokens
            (r"sk-[A-Za-z0-9]{48}", "<REDACTED_API_KEY>"),  # OpenAI-style API keys
            (r"password[=:]\s*[^\s]+", "password=<REDACTED>"),  # Password assignments
            (r"token[=:]\s*[^\s]+", "token=<REDACTED>"),  # Token assignments
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
    
    
if __name__ == "__main__":
    from openevolve_graph.Config.config import Config
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    sampler = PromptSampler_langchain(config.prompt)
    test_code ='''
    def add(a,b):
        return a+b
    '''
    prompt = sampler.build_prompt(current_program=test_code,template_key="evaluation")
    # ##logger.info("prompt","\n",prompt)