from typing import Any, Dict, Union
from classes import MessageList, SamplerBase
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


class HFChatCompletionSampler(SamplerBase):
    def __init__(
        self,
        model: str,
        model_dir: Union[str, None] = None,
        API_TOKEN: Union[str, None] = os.environ.get("HF_TOKEN", None),
        system_message: Union[str, None] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        if model_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _pack_message(self, role: str, content: Any) -> Dict:
        return {"role": str(role), "content": str(content)}

    def _pack_message_to_string(self, message_list: MessageList) -> str:
        prompt = ""
        for msg in message_list:
            if msg["role"] == "system":
                prompt += f"{msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\nAssistant: "
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']}\n"

        return prompt

    def __call__(self, message_list: MessageList) -> str:
        # Format messages into prompt
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list

        prompt = self._pack_message_to_string(message_list)

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        try:
            return response.split("Assistant: ")[-1].strip()
        except Exception:
            return response
