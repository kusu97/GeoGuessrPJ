from models.base import BaseModel
from utils.parser import parse_response
from openai import OpenAI
import os
import base64
import json
import time

class QwenModel(BaseModel):
    """
    class for Qwen-vl model
    """

    def __init__(self, model_name="qwen3-vl-plus", enable_thinking=False, save_responses=True):
        self.enable_thinking = enable_thinking
        self.model_name = model_name
        self.save_responses = save_responses
        
        self.client = OpenAI(
            api_key= os.getenv("DASHSCOPE_API_KEY"),
            base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def chat_with_mllm(self, image_base64, media_type, prompt):
        # 创建聊天完成请求
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            stream=False,
            # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
            extra_body={
                'enable_thinking': self.enable_thinking,
                "thinking_budget": 81920},
        )

        message = completion.choices[0].message
        usage = completion.usage

        answer_content = message.content
        reasoning_content = getattr(message, "reasoning_content", "")

        return {"reasoning_content": reasoning_content,
                "answer_content": answer_content, 
                "usage": usage}

    def chat_with_mllm_streaming(self, image_base64, media_type, prompt):
        '''Discarded, since the streaming functionality is not needed.'''
        reasoning_content = ""  # 定义完整思考过程
        answer_content = ""     # 定义完整回复
        is_answering = False   # 判断是否结束思考过程并开始回复
        
        # 创建聊天完成请求
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            stream=True,
            # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
            extra_body={
                'enable_thinking': self.enable_thinking,
                "thinking_budget": 81920},
            # 在最后一个chunk返回Token使用量
            stream_options={
                "include_usage": True
            }
        )

        if self.enable_thinking:
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            # 如果chunk.choices为空，则打印usage
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    # 开始回复
                    if delta.content != "" and is_answering is False:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    # 打印回复过程
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content

        return {"reasoning_content": reasoning_content,
                "answer_content": answer_content}

    def encode_image_to_base64(self, image_path):
        media_type = self.get_image_media_type(image_path)
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        return img_data, media_type
    
    def get_image_media_type(self, image_path: str) -> str:
        """Determine the media type based on file extension."""
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.png':
            return "image/png"
        elif ext in ['.jpg', '.jpeg']:
            return "image/jpeg"
        else:
            return "image/jpeg" # Default fallback

    def save_response(self, prompt: str, image_path: str, response: dict, save_dir: str = "./records/responses"):
        """
        Save model response to a JSON file.
        
        If two calls occur within one second, an overwrite error may happen; 
        however, at this stage, this remains only a theoretical risk.
        """
        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.json"
        save_path = os.path.join(save_dir, filename)

        # 将 usage 转成字典，如果是 None 则保留 None
        usage_dict = None
        usage = response.get("usage", None)
        if usage:
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            completion_details = getattr(usage, "completion_tokens_details", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),

                "prompt_image_tokens": getattr(prompt_details, "image_tokens", None),
                "prompt_text_tokens": getattr(prompt_details, "text_tokens", None),

                "completion_text_tokens": getattr(completion_details, "text_tokens", None),
                "completion_reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
            }

        data = {
            "model_name": self.model_name,
            "enable_thinking": self.enable_thinking,
            "prompt": prompt,
            "image_path": image_path,
            "reasoning_content": response.get("reasoning_content", ""),
            "answer_content": response.get("answer_content", ""),
            "usage": usage_dict,
            "timestamp": timestamp,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return save_path
    
    def predict(self, image_path: str, prompt: str):
        image_base64, media_type = self.encode_image_to_base64(image_path)
        response = self.chat_with_mllm(image_base64, media_type, prompt)
        if self.save_responses:
            self.save_response(prompt, image_path, response)
        answer_content = response["answer_content"]
        pred = parse_response(answer_content)
        return pred


if __name__ == '__main__':
    model = QwenModel("qwen-vl-max-2025-04-08")