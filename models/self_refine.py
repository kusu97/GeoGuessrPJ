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
from models.agents.refine_agent import create_refine_agent


class SelfRefine(BaseModel):
    '''
    Class for a self-refine pipeline implemented using AutoGen.

    This pipeline implements a Self-Refine reasoning process for the
    image geolocation task using a single agent with iterative
    self-critique and refinement.

    The agent first produces an initial reasoning and location
    prediction based on the input image. 
    
    Then it critically evaluates its own reasoning to identify potential 
    errors, weak assumptions or overlooked visual evidence.

    Using this self-generated feedback, the agent refines and improves
    its reasoning and prediction. 
    
    This process is repeated for several iterations (typically 2 rounds) 
    to progressively enhance the quality and consistency of the 
    geolocation inference.

    The reasoning flow follows an iterative refinement loop:

        Image
        ↓
        Initial Reasoning
        ↓
        Self-Critique (Feedback)
        ↓
        Refined Reasoning
        ↓
        ... (repeat for several iterations)
        ↓
        Final Prediction
        (country, latitude, longitude)
    '''

    def __init__(self, refine_client_info, save_responses=True):

        self.pipeline_name = "self-refine"
        self.save_responses = save_responses

        self.refine_client_info = refine_client_info
        self.all_client_info = {
            "refine": refine_client_info,
        }

        self.refine_model_client = OpenAIChatCompletionClient(**self.refine_client_info)

        self.refine_agent = create_refine_agent(self.refine_model_client)

        self.cancellation_token = CancellationToken()

        self.init_prompt = """
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
        self.critique_prompt = """
Now you are required to review your previous reasoning and prediction.

You should critically evaluate your own reasoning and identify potential problems, such as:

- incorrect interpretation of visual clues
- missing evidence
- overconfident assumptions
- inconsistencies between clues
- alternative plausible locations that were ignored

Based on your evaluation and identification, provide a structured critique of your previous reasoning, including the suggestions for improving the reasoning.

Your output format MUST adhere to the following format:

Feedback: <your critique>
"""
        self.refinement_prompt = """
Now you are required to refine and improve your reasoning and prediction based on the feedback which you just generated.

Requirements:

- Correct the mistakes and weaknesses identified in the critique
- Produce clearer and more reliable reasoning and prediction
- Assign a precise country and coordinates in your prediction, even if you may be not sure about it

Your output format MUST adhere to the following format:

Reasoning: <step-by-step geographic reasoning>

Prediction:
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

        # ---------- Step 1 Initial Reasoning and Prediction ----------

        pil_image = PILImage.open(image_path)

        init_message = MultiModalMessage(
            content=[
                self.init_prompt,
                Image(pil_image),
            ],
            source="user",
        )

        init_response = await self.refine_agent.on_messages([init_message], 
                                                    cancellation_token=self.cancellation_token)

        init_output = init_response.chat_message.content
        init_usage = self._get_usage(init_response.chat_message.models_usage)


        # ---------- Step 2 Iterative Self-Refinement ----------

        critique_message = TextMessage(
            content=self.critique_prompt,
            source="user",
        )

        refinement_message = TextMessage(
            content=self.refinement_prompt,
            source="user",
        )
        
        # Round 1

        critique_round_1_response = await self.refine_agent.on_messages([critique_message], 
                                            cancellation_token = self.cancellation_token)

        critique_round_1_output = critique_round_1_response.chat_message.content
        critique_round_1_usage = self._get_usage(critique_round_1_response.chat_message.models_usage)

        refinement_round_1_response = await self.refine_agent.on_messages([refinement_message], 
                                                    cancellation_token=self.cancellation_token)

        refinement_round_1_output = refinement_round_1_response.chat_message.content
        refinement_round_1_usage = self._get_usage(refinement_round_1_response.chat_message.models_usage)

        # Round 2

        critique_round_2_response = await self.refine_agent.on_messages([critique_message], 
                                            cancellation_token = self.cancellation_token)

        critique_round_2_output = critique_round_2_response.chat_message.content
        critique_round_2_usage = self._get_usage(critique_round_2_response.chat_message.models_usage)

        refinement_round_2_response = await self.refine_agent.on_messages([refinement_message], 
                                                    cancellation_token=self.cancellation_token)

        refinement_round_2_output = refinement_round_2_response.chat_message.content
        refinement_round_2_usage = self._get_usage(refinement_round_2_response.chat_message.models_usage)


        return ((init_output, critique_round_1_output, refinement_round_1_output, critique_round_2_output, refinement_round_2_output), 
                (init_usage, critique_round_1_usage, refinement_round_1_usage, critique_round_2_usage, refinement_round_2_usage))

    def run_sync(self, image_path):
        '''The serial version of the self.run()'''
        return asyncio.run(self.run(image_path))
    
    async def reset(self):
        """
        Reset all the assistant agents to the initialization state to clear memories.
        
        To ensure independence between samples during benchmarking, all agent 
        memories are reset before processing each image.
        """
        await self.refine_agent.on_reset(self.cancellation_token)

    async def predict(self, image_path: str, prompt: str):
        # Note: prompt not needed
        outputs = await self.run(image_path)
        if self.save_responses:
            self.save_response(image_path, outputs)
        final_response = outputs[0][4]
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
            "refine_client_info": self.refine_client_info,
            "image_path": image_path,
            "outputs": {
                "init_output": outputs[0][0], 
                "critique_round_1_output": outputs[0][1], 
                "refinement_round_1_output": outputs[0][2], 
                "critique_round_2_output": outputs[0][3], 
                "refinement_round_2_output (final_output)": outputs[0][4]
            },
            "usage": {
                "init_usage": outputs[1][0],
                "critique_round_1_usage": outputs[1][1],
                "refinement_round_1_usage": outputs[1][2],
                "critique_round_2_usage": outputs[1][3],
                "refinement_round_2_usage": outputs[1][4]
            },
            "timestamp": timestamp
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return save_path

if __name__ == "__main__":
    pass
