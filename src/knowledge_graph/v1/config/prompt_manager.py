"""
YAML Prompt Manager packaged under knowledge_graph.v1.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class PromptConfigError(Exception):
    pass


class YAMLPromptManager:
    def __init__(self, config_file: Optional[Path] = None):
        if config_file is None:
            config_file = Path(__file__).parent / "prompts.yaml"
        self.config_file = Path(config_file)
        self._prompts: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        try:
            if not self.config_file.exists():
                raise PromptConfigError(f"Prompt configuration file not found: {self.config_file}")
            with open(self.config_file, 'r', encoding='utf-8') as file:
                self._prompts = yaml.safe_load(file) or {}
            logger.debug(f"Successfully loaded prompts from {self.config_file}")
        except yaml.YAMLError as e:
            raise PromptConfigError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise PromptConfigError(f"Error loading prompt configuration: {e}")

    def get_prompt(self, category: str, prompt_type: str = "system") -> str:
        try:
            return self._prompts[category][prompt_type]
        except KeyError:
            available_categories = list(self._prompts.keys())
            raise PromptConfigError(
                f"Prompt '{category}:{prompt_type}' not found. Available categories: {available_categories}"
            )

    def get_formatted_prompt(self, category: str, prompt_type: str = "user", **kwargs) -> str:
        template = self.get_prompt(category, prompt_type)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise PromptConfigError(
                f"Missing placeholder value for: {e}. Provided: {list(kwargs.keys())}"
            )

    def get_extraction_system_prompt(self) -> str:
        return self.get_prompt('triplets_generator', 'system')

    def get_extraction_user_prompt(self, text: str) -> str:
        return self.get_formatted_prompt('triplets_generator', 'user', text=text)

    def get_merge_system_prompt(self) -> str:
        return self.get_prompt('merging', 'system')

    def get_merge_user_prompt(self, entity_name: str, existing_content: str, new_content: str) -> str:
        return self.get_formatted_prompt('merging', 'user', entity_name=entity_name, existing_content=existing_content, new_content=new_content)

    def get_retrieval_response_system_prompt(self) -> str:
        return self.get_prompt('Retrieval_response', 'system')

    def get_summary_system_prompt(self) -> str:
        return self.get_prompt('summary', 'system')

    def get_summary_user_prompt(self, root_node_name: str, combined_children_content: str) -> str:
        return self.get_formatted_prompt('summary', 'user', root_node_name=root_node_name, combined_children_content=combined_children_content)

    def get_retrieval_response_user_prompt(self, query, content) -> str:
        return self.get_formatted_prompt('Retrieval_response', 'user', query=query, content=content)

    def get_context_relevance_user_prompt(self, query: str, context: str) -> str:
        return self.get_formatted_prompt('context_relevance', 'user', query=query, context=context)

    def get_initial_summarisation_prompt(self, instruction: str, document: str) -> str:
        return self.get_formatted_prompt('initial_summarisation', 'user', instruction=instruction, document=document)

    def reload_prompts(self) -> None:
        logger.debug("Reloading prompt configuration...")
        self._load_prompts()


prompt_manager = YAMLPromptManager()


