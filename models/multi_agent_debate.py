import os
import json
import time
import asyncio
from PIL import Image as PILImage
from dataclasses import asdict

from autogen_core import Image, CancellationToken
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from models.base import BaseModel
from utils.parser import parse_response_with_exception_handler
from models.agents.debater import create_debater
from models.agents.judge_agent import create_judge_agent


class MultiAgentDebate(BaseModel):
    '''
    Class for a multi-agent debate pipeline implemented using AutoGen.

    This pipeline approaches the image geolocation task through a structured
    multi-agent debate process. It consists of two types of agents:

    1. DebaterAgents
    Two geographic reasoning agents independently analyze the visual
    observations extracted from the image and propose initial location
    predictions. Both agents share the same reasoning framework but may
    produce different hypotheses.

    The debaters then engage in multiple rounds of argumentation, where
    each agent evaluates the opponent's reasoning, identifies potential
    weaknesses, and may revise its own prediction accordingly.

    2. JudgeAgent
    After several rounds of debate, a judge agent evaluates the final
    arguments produced by the two debaters.

    If both agents converge to the same prediction, the consensus result
    is accepted. If their predictions differ, the judge selects the answer
    supported by the more convincing reasoning.

    The debate process follows the structure:

        Image
        ↓
        Debater A ↔ Debater B
        (3 rounds of debate)
        ↓
        JudgeAgent
        ↓
        (country, latitude, longitude)
    
    Note: Different temperature of each agent is encouraged.
        e.g. DebaterA temperature = 0.8
            DebaterB temperature = 1.0
            Judge temperature = 0.2
    '''

    def __init__(self, debaterA_client_info, debaterB_client_info, judge_client_info, save_responses=True):

        self.pipeline_name = "multi-agent debate"
        self.save_responses = save_responses

        self.debaterA_client_info = debaterA_client_info
        self.debaterB_client_info = debaterB_client_info
        self.judge_client_info = judge_client_info
        self.all_client_info = {
            "debaterA": debaterA_client_info,
            "debaterB": debaterB_client_info,
            "judge": judge_client_info
        }

        self.debaterA_model_client = OpenAIChatCompletionClient(**self.debaterA_client_info)
        self.debaterB_model_client = OpenAIChatCompletionClient(**self.debaterB_client_info)
        self.judge_model_client = OpenAIChatCompletionClient(**self.judge_client_info)

        self.debaterA = create_debater(self.debaterA_model_client)
        self.debaterB = create_debater(self.debaterB_model_client)
        self.judge_agent = create_judge_agent(self.judge_model_client)

        self.cancellation_token = CancellationToken()

        self.debaterA_init_prompt = """
You are the first debater in a geolocation task.

You are given an image.

You are required to extract visual observations from the image and analyze these observations carefully to infer the most likely geographic location.

You can use any visible clues such as:

- signage and language
- vegetation and climate
- architecture and building style
- terrain and landscape
- infrastructure and urban layout

Explain your reasoning step by step before producing your prediction.

Your output format MUST adhere to the following format:

Reasoning: <step-by-step geographic reasoning>

Prediction:
country: <country name>
lat: <latitude as a decimal number>
lng: <longitude as a decimal number>
"""
        self.debaterB_init_prompt = """
You are the second debater in a geolocation task.

You are given an image.

You are required to independently infer the most likely geographic location based on the visual clues extracted from the image.

Your reasoning should be concise but precise. Focus on identifying the strongest geographic signals.

Your output format MUST adhere to the following format:

Reasoning: <geographic reasoning>

Prediction:
country: <country name>
lat: <latitude as a decimal number>
lng: <longitude as a decimal number>
"""
        self.debate_round_1_prompt = """
Now this is the first round of the geolocation debate.

The following is the opposing debater's answer:

{}

You are required to revise your previous argument based on other agent's reasoning.

If you believe your original conclusion is still correct, you can also defend it with stronger evidence.

Your output format MUST adhere to the following format:

Updated Reasoning: <geographic reasoning>

Updated Prediction:
country: <country name>
lat: <latitude as a decimal number>
lng: <longitude as a decimal number>
"""
        self.debate_round_2_prompt = """
Now this is the second round of the geolocation debate.

The following is the opponent's latest answer:

{}

You are required to revise your previous argument based on other agent's reasoning.

If you believe your previous conclusion is still correct, you can also defend it with stronger evidence.

Your output format MUST adhere to the following format:

Updated Reasoning: <geographic reasoning>

Updated Prediction:
country: <country name>
lat: <latitude as a decimal number>
lng: <longitude as a decimal number>
"""
        self.debate_final_round_prompt = """
Now this is the final round of the geolocation debate.

The following is the opponent's latest answer:

{}

You are required to revise your previous argument based on other agent's reasoning.

If you believe your previous conclusion is still correct, you can also defend it with stronger evidence.

In your final output, you must produce the final answer together with a concise reasoning content, which must be a complete reasoning summarizing the final conclusions you've reached after the debate.

Your output format MUST adhere to the following format:

Final Reasoning: <geographic reasoning>

Final Prediction:
country: <country name>
lat: <latitude as a decimal number>
lng: <longitude as a decimal number>
"""
        self.judge_prompt = """
You are given an image.

Below are the final reasoning and predictions produced by the two debaters about the geographic location in the image.

Debater A Final Answer:

{}

Debater B Final Answer:

{}

You are required to determine the most reliable geographic prediction among these two answers.

If they agree with each other, confirm the consensus result; If they disagree, evaluate both arguments and select the more plausible prediction.

Your output format MUST adhere to the following format:

Reasoning Summary: <short explanation of why the chosen answer is more reliable>

Final Prediction:
country: <country name>
lat: <latitude as a decimal number>
lng: <longitude as a decimal number>
"""

    def _get_usage(self, models_usage):
        if models_usage is None:
            return None
        else:
            return asdict(models_usage)

    async def run(self, image_path):

        pil_image = PILImage.open(image_path)

        # ---------- Step 1 Initial Reasoning ----------

        debaterA_init_message = MultiModalMessage(
            content=[
                self.debaterA_init_prompt,
                Image(pil_image),
            ],
            source="user",
        )

        debaterB_init_message = MultiModalMessage(
            content=[
                self.debaterB_init_prompt,
                Image(pil_image),
            ],
            source="user",
        )

        debaterA_init_response = await self.debaterA.on_messages([debaterA_init_message], 
                                                    cancellation_token=self.cancellation_token)
        debaterB_init_response = await self.debaterB.on_messages([debaterB_init_message], 
                                                    cancellation_token=self.cancellation_token)

        debaterA_init_output = debaterA_init_response.chat_message.content
        debaterA_init_usage = self._get_usage(debaterA_init_response.chat_message.models_usage)
        debaterB_init_output = debaterB_init_response.chat_message.content
        debaterB_init_usage = self._get_usage(debaterB_init_response.chat_message.models_usage)

        # ---------- Step 2 Debate ----------

        # Round 1
        debaterA_round_1_message = TextMessage(
            content=self.debate_round_1_prompt.format(debaterB_init_output),
            source="user",
        )
        debaterB_round_1_message = TextMessage(
            content=self.debate_round_1_prompt.format(debaterA_init_output),
            source="user",
        )

        debaterA_round_1_response = await self.debaterA.on_messages([debaterA_round_1_message], 
                                            cancellation_token = self.cancellation_token)
        debaterB_round_1_response = await self.debaterB.on_messages([debaterB_round_1_message], 
                                            cancellation_token = self.cancellation_token)

        debaterA_round_1_output = debaterA_round_1_response.chat_message.content
        debaterA_round_1_usage = self._get_usage(debaterA_round_1_response.chat_message.models_usage)
        debaterB_round_1_output = debaterB_round_1_response.chat_message.content
        debaterB_round_1_usage = self._get_usage(debaterB_round_1_response.chat_message.models_usage)

        # Round 2
        debaterA_round_2_message = TextMessage(
            content=self.debate_round_2_prompt.format(debaterB_round_1_output),
            source="user",
        )
        debaterB_round_2_message = TextMessage(
            content=self.debate_round_2_prompt.format(debaterA_round_1_output),
            source="user",
        )

        debaterA_round_2_response = await self.debaterA.on_messages([debaterA_round_2_message], 
                                            cancellation_token = self.cancellation_token)
        debaterB_round_2_response = await self.debaterB.on_messages([debaterB_round_2_message], 
                                            cancellation_token = self.cancellation_token)

        debaterA_round_2_output = debaterA_round_2_response.chat_message.content
        debaterA_round_2_usage = self._get_usage(debaterA_round_2_response.chat_message.models_usage)
        debaterB_round_2_output = debaterB_round_2_response.chat_message.content
        debaterB_round_2_usage = self._get_usage(debaterB_round_2_response.chat_message.models_usage)

        # Final Round
        debaterA_final_round_message = TextMessage(
            content=self.debate_final_round_prompt.format(debaterB_round_2_output),
            source="user",
        )
        debaterB_final_round_message = TextMessage(
            content=self.debate_final_round_prompt.format(debaterA_round_2_output),
            source="user",
        )

        debaterA_final_round_response = await self.debaterA.on_messages([debaterA_final_round_message], 
                                            cancellation_token = self.cancellation_token)
        debaterB_final_round_response = await self.debaterB.on_messages([debaterB_final_round_message], 
                                            cancellation_token = self.cancellation_token)

        debaterA_final_round_output = debaterA_final_round_response.chat_message.content
        debaterA_final_round_usage = self._get_usage(debaterA_final_round_response.chat_message.models_usage)
        debaterB_final_round_output = debaterB_final_round_response.chat_message.content
        debaterB_final_round_usage = self._get_usage(debaterB_final_round_response.chat_message.models_usage)

        # ---------- Step 3 Final Judgement ----------

        judge_message = MultiModalMessage(
            content=[
                self.judge_prompt.format(debaterA_final_round_output, debaterB_final_round_output),
                Image(pil_image),
            ],
            source="user",
        )

        judge_response = await self.judge_agent.on_messages([judge_message], 
                                                    cancellation_token=self.cancellation_token)

        final_output = judge_response.chat_message.content
        judge_usage = self._get_usage(judge_response.chat_message.models_usage)

        return ((debaterA_init_output, debaterB_init_output, debaterA_round_1_output, 
                 debaterB_round_1_output, debaterA_round_2_output, debaterB_round_2_output, 
                 debaterA_final_round_output, debaterB_final_round_output, final_output), 
                (debaterA_init_usage, debaterB_init_usage, debaterA_round_1_usage, 
                 debaterB_round_1_usage, debaterA_round_2_usage, debaterB_round_2_usage, 
                 debaterA_final_round_usage, debaterB_final_round_usage, judge_usage))

    def run_sync(self, image_path):
        '''The serial version of the self.run()'''
        return asyncio.run(self.run(image_path))
    
    async def reset(self):
        """
        Reset all the assistant agents to the initialization state to clear memories.
        
        To ensure independence between samples during benchmarking, all agent 
        memories are reset before processing each image.
        """
        await asyncio.gather(
            self.debaterA.on_reset(self.cancellation_token),
            self.debaterB.on_reset(self.cancellation_token),
            self.judge_agent.on_reset(self.cancellation_token),
        )

    async def predict(self, image_path: str, prompt: str):
        # Note: prompt not needed
        outputs = await self.run(image_path)
        if self.save_responses:
            self.save_response(image_path, outputs)
        final_response = outputs[0][8]
        pred = parse_response_with_exception_handler(final_response)    # pred = None if parse_response fails
        return pred
    
    def save_response(self, image_path: str, outputs: tuple[str], save_dir: str = "./records/responses"):
        """
        Save model outputs to a JSON file.
        
        If two calls occur within one second, an overwrite error may happen; 
        however, at this stage, this remains only a theoretical risk.
        """
        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.json"
        save_path = os.path.join(save_dir, filename)

        data = {
            "pipeline": self.pipeline_name,
            "debaterA_client_info": self.debaterA_client_info,
            "debaterB_client_info": self.debaterB_client_info,
            "judge_client_info": self.judge_client_info,
            "image_path": image_path,
            "debate_outputs": {
                "debaterA_init_output": outputs[0][0],
                "debaterB_init_output": outputs[0][1],
                "debaterA_round_1_output": outputs[0][2],
                "debaterB_round_1_output": outputs[0][3],
                "debaterA_round_2_output": outputs[0][4],
                "debaterB_round_2_output": outputs[0][5],
                "debaterA_final_round_output": outputs[0][6],
                "debaterB_final_round_output": outputs[0][7]
            },
            "final_output": outputs[0][8],
            "usage": {
                "debaterA_init_usage": outputs[1][0],
                "debaterB_init_usage": outputs[1][1],
                "debaterA_round_1_usage": outputs[1][2],
                "debaterB_round_1_usage": outputs[1][3],
                "debaterA_round_2_usage": outputs[1][4],
                "debaterB_round_2_usage": outputs[1][5],
                "debaterA_final_round_usage": outputs[1][6],
                "debaterB_final_round_usage": outputs[1][7],
                "judge_usage": outputs[1][8]
            },
            "timestamp": timestamp
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return save_path

if __name__ == "__main__":
    pass