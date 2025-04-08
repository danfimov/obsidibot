import abc

from openai import OpenAI


class AbstractLLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass


class NebiusLLMProvider(AbstractLLMProvider):
    def __init__(
        self,
        api_token: str,
    ) -> None:
        self._client = OpenAI(
            base_url='https://api.studio.nebius.com/v1/',
            api_key=api_token,
        )

    def generate_response(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model='deepseek-ai/DeepSeek-R1',
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            messages=[{'role': 'user', 'content': prompt}],
        )
        response_message = response.choices[0].message.content
        if not response_message:
            return ''
        return response_message
