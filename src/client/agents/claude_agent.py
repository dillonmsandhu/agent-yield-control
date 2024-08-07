import anthropic
import os
from copy import deepcopy
from typing import List

from ..agent import AgentClient


class Claude(AgentClient):
    def __init__(self, api_args=None, *args, **config):
        super().__init__(*args, **config)
        if not api_args:
            api_args = {}
        api_args = deepcopy(api_args)
        print("API Args" , api_args)
        self.key = api_args.pop("key", None) or os.getenv('Claude_API_KEY')
        api_args["model"] = api_args.pop("model", None)
        if not self.key:
            raise ValueError("Claude API KEY is required, please assign api_args.key or set OPENAI_API_KEY "
                             "environment variable.")
        if not api_args["model"]:
            raise ValueError("Claude model is required, please assign api_args.model.")
        self.api_args = api_args
        if not self.api_args.get("stop_sequences"):
            self.api_args["stop_sequences"] = [anthropic.HUMAN_PROMPT]

    def inference(self, history: List[dict]) -> str:
        c = anthropic.Anthropic(api_key=self.key)
        claude_role_hist = [ {'role': 'assistant' if h['role'] == 'agent' else 'user', 'content': h['content']} for h in history]
        message = c.messages.create(messages=claude_role_hist, **self.api_args) # requires model, max_tokens, and messages
        return message.content[0].text
