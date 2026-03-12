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
from models.agents.vision_agent import create_vision_agent
from models.agents.geo_agent import create_geo_agent
from models.agents.localization_agent import create_localization_agent


class MultiAgentCollaboration(BaseModel):
    '''
    Class for a multi-agent collaboration pipeline implemented using AutoGen.

    This pipeline decomposes the image geolocation task into three
    collaborating agents:

    1. VisionAgent
    Analyzes the input image and extracts structured visual
    observations relevant to geographic localization. The output
    is returned as a JSON object describing visual clues such as
    language, vegetation, climate, or architectural style.

    2. GeoReasoningAgent
    Consumes the visual observations and performs geographic
    reasoning based on world knowledge (e.g., climate zones,
    language distribution, infrastructure patterns) to infer
    possible regions or countries.

    3. LocalizationAgent
    Produces the final geolocation prediction by estimating the
    most likely country and geographic coordinates based on the
    reasoning provided by the previous agent.

    The agents collaborate sequentially in a pipeline:

        Image
        ↓
        VisionAgent
        ↓
        GeoReasoningAgent
        ↓
        LocalizationAgent
        ↓
        (country, latitude, longitude)
    '''

    def __init__(self, vision_client_info, geo_client_info, localization_client_info, save_responses=True):

        self.pipeline_name = "multi-agent collaboration"
        self.save_responses = save_responses

        self.vision_client_info = vision_client_info
        self.geo_client_info = geo_client_info
        self.localization_client_info = localization_client_info
        self.all_client_info = {
            "vision": vision_client_info,
            "geo_reasoning": geo_client_info,
            "localization": localization_client_info
        }

        self.vision_model_client = OpenAIChatCompletionClient(**self.vision_client_info)
        self.geo_model_client = OpenAIChatCompletionClient(**self.geo_client_info)
        self.localization_model_client = OpenAIChatCompletionClient(**self.localization_client_info)

        self.vision_agent = create_vision_agent(self.vision_model_client)
        self.geo_agent = create_geo_agent(self.geo_model_client)
        self.localization_agent = create_localization_agent(self.localization_model_client)

        self.cancellation_token = CancellationToken()

        self.vision_prompt = """
You are given an image.

Your task is to extract visual observations that may help determine the geographic location.

Identify any visible clues such as:

- signage and language
- vegetation and climate
- architecture and building style
- terrain and landscape
- infrastructure and urban layout

Return the observations as a JSON object.

Each key should represent a type of visual clue, and each value should describe the observation.

Example format:

{
  "language": "Cyrillic",
  "vegetation": "conifer forest",
  "architecture": "Soviet-style apartment buildings"
}

Include as many relevant clues as you can identify.
If no clues are visible, return an empty JSON object.

Return ONLY a valid JSON object.
Do not include explanations or text outside the JSON.
"""
        self.geo_prompt = """
You are given visual observations extracted from an image.

The observations are provided as a JSON object where each key represents a type of visual clue and each value describes the observation.

Visual observations:

{}

Use these vision clues to infer possible geographic regions.

Reason step-by-step about how these clues support or contradict possible regions.

Explain which regions are most likely and why.
"""
        self.localization_prompt = """
You are given geographic reasoning about a location.

Reasoning:

{}

Based on this reasoning, estimate the most likely country and approximate latitude and longitude.


Your final answer MUST include these three lines somewhere in your response:

country: [country name]
lat: [latitude as a decimal number]
lng: [longitude as a decimal number]
"""

    def _get_usage(self, models_usage):
        if models_usage is None:
            return None
        else:
            return asdict(models_usage)

    async def run(self, image_path):

        # ---------- Step 1 Vision Clues Extracting ----------

        pil_image = PILImage.open(image_path)

        vision_message = MultiModalMessage(
            content=[
                self.vision_prompt,
                Image(pil_image),
            ],
            source="user",
        )

        vision_response = await self.vision_agent.on_messages([vision_message], 
                                                    cancellation_token=self.cancellation_token)

        vision_output = vision_response.chat_message.content
        vision_usage = self._get_usage(vision_response.chat_message.models_usage)


        # ---------- Step 2 Geo Reasoning ----------

        geo_message = TextMessage(
            content=self.geo_prompt.format(vision_output),
            source="user",
        )

        geo_response = await self.geo_agent.on_messages([geo_message], 
                                            cancellation_token = self.cancellation_token)

        geo_output = geo_response.chat_message.content
        geo_usage = self._get_usage(geo_response.chat_message.models_usage)


        # ---------- Step 3 Localization ----------

        loc_message = TextMessage(
            content=self.localization_prompt.format(geo_output),
            source="user",
        )

        loc_response = await self.localization_agent.on_messages([loc_message], 
                                                    cancellation_token=self.cancellation_token)

        final_output = loc_response.chat_message.content
        loc_usage = self._get_usage(loc_response.chat_message.models_usage)

        return ((vision_output, geo_output, final_output), 
                (vision_usage, geo_usage, loc_usage))

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
            self.vision_agent.on_reset(self.cancellation_token),
            self.geo_agent.on_reset(self.cancellation_token),
            self.localization_agent.on_reset(self.cancellation_token),
        )

    async def predict(self, image_path: str, prompt: str):
        # Note: prompt not needed
        outputs = await self.run(image_path)
        if self.save_responses:
            self.save_response(image_path, outputs)
        final_response = outputs[0][2]
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
            "vision_client_info": self.vision_client_info,
            "geo_client_info": self.geo_client_info,
            "localization_client_info": self.localization_client_info,
            "image_path": image_path,
            "vision_output": outputs[0][0],
            "geo_reasoning_output": outputs[0][1],
            "final_output": outputs[0][2],
            "usage": {
                "vision_usage": outputs[1][0],
                "geo_reasoning_usage": outputs[1][1],
                "localization_usage": outputs[1][2]
            },
            "timestamp": timestamp
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return save_path

if __name__ == "__main__":
    pass