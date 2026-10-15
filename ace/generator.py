"""
Generator agent for ACE system.
Generates answers to questions using playbook and reflection.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any

from .llm import timed_llm_call


class Generator:
    """
    Generator agent that produces answers to questions using knowledge
    from a playbook and previous reflections.
    """

    def __init__(self, api_client, model: str, prompt_template: str, uses_context: bool = False,
                 max_tokens: int = 4096, temperature: float = 0.7):
        """
        Args:
            api_client: OpenAI client for LLM calls
            model: Model name to use for generation
            prompt_template: task generator prompt (positional slots: playbook, reflection, question[, context])
            uses_context: whether the prompt has a context slot (HotpotQA)
            max_tokens: Maximum tokens for generation
        """
        self.api_client = api_client
        self.model = model
        self.prompt_template = prompt_template
        self.uses_context = uses_context
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate an answer to a question using the playbook.

        Returns:
            Tuple of (full_response, bullet_ids_used, call_info)
        """
        # Keep the prompt structure available so the LLM layer can truncate
        # only the playbook while preserving the instructions, question, and context.
        if self.uses_context:
            def prompt_builder(playbook_content: str) -> str:
                return self.prompt_template.format(playbook_content, reflection, question, context)
        else:
            def prompt_builder(playbook_content: str) -> str:
                return self.prompt_template.format(playbook_content, reflection, question)

        prompt = prompt_builder(playbook)

        response, call_info = timed_llm_call(
            self.api_client,
            self.model,
            prompt,
            role="generator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.temperature,
            playbook=playbook,
            prompt_builder=prompt_builder,
        )

        bullet_ids = self._extract_bullet_ids(response, use_json_mode)

        return response, bullet_ids, call_info

    def _extract_bullet_ids(self, response: str, use_json_mode: bool) -> List[str]:
        """Extract bullet IDs from generator response."""
        bullet_ids = []

        if use_json_mode:
            try:
                response_json = json.loads(response)
                bullet_ids = response_json.get("bullet_ids", [])
            except (json.JSONDecodeError, KeyError):
                # If parsing fails, try regex extraction
                bullet_ids = self._extract_bullet_ids_regex(response)
        else:
            bullet_ids = self._extract_bullet_ids_regex(response)

        return bullet_ids

    def _extract_bullet_ids_regex(self, text: str) -> List[str]:
        """Extract bullet IDs using regex pattern matching."""
        # Pattern matches: [xxx-00001], [abc-00042], etc.
        pattern = r'\[([a-z]{3,}-\d{5})\]'
        matches = re.findall(pattern, text)
        return matches
