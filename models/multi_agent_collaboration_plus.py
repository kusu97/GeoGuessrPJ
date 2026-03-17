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
from models.agents.visual_analyst import create_visual_analyst
from models.agents.geolocation_agent import create_geolocation_agent


class MultiAgentCollaborationPlus(BaseModel):
    '''
    Class for an updated multi-agent collaboration pipeline implemented using AutoGen.

    This pipeline decomposes the image geolocation task into two
    collaborating agents:

    1. VisualAnalystAgent
    Analyzes the input image and extracts visual observations 
    relevant to geographic localization.

    2. GeolocationAgent
    Consumes the visual observations and performs geographic
    localization based on world knowledge to infer the most 
    likely country and geographic coordinates.

    The agents collaborate sequentially in a pipeline:

        Image
        ↓
        VisualAnalystAgent
        ↓
        GeolocationAgent
        ↓
        (country, latitude, longitude)
    '''

    def __init__(self, visual_anaylst_client_info, geolocation_client_info, save_responses=True):

        self.pipeline_name = "multi-agent collaboration plus"
        self.save_responses = save_responses

        self.visual_analyst_client_info = visual_anaylst_client_info
        self.geolocation_client_info = geolocation_client_info
        self.all_client_info = {
            "visual_analyst": visual_anaylst_client_info,
            "geolocation": geolocation_client_info
        }

        self.visual_analyst_model_client = OpenAIChatCompletionClient(**self.visual_analyst_client_info)
        self.geolocation_model_client = OpenAIChatCompletionClient(**self.geolocation_client_info)

        self.visual_analyst = create_visual_analyst(self.visual_analyst_model_client)
        self.geolocation_agent = create_geolocation_agent(self.geolocation_model_client)

        self.cancellation_token = CancellationToken()

        self.visual_analyst_prompt = """
You are given a street view image.

Based on the provided image, you are required to carefully analyze specific visual clues relevant to geolocation, including but not limited to:
   - signage and language
   - vegetation and climate
   - architecture and building style
   - terrain and landscape
   - infrastructure and urban layout

Describe the visual observations you identify in detail.

In your output, you should include as many relevant clues as you can find.
"""
        self.geolocation_prompt = """
You are given detailed visual observations extracted from a street view image.

The following is the visual observations:

{}

Use these vision clues to infer the possible geographic location.

Reason step-by-step to estimate the most likely country and approximate latitude and longitude.

Your final answer MUST include these three lines somewhere in your output:

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

        # ---------- Step 1 Visual Clues Extracting ----------

        pil_image = PILImage.open(image_path)

        visual_message = MultiModalMessage(
            content=[
                self.visual_analyst_prompt,
                Image(pil_image),
            ],
            source="user",
        )

        visual_response = await self.visual_analyst.on_messages([visual_message], 
                                                    cancellation_token=self.cancellation_token)

        visual_output = visual_response.chat_message.content
        visual_usage = self._get_usage(visual_response.chat_message.models_usage)


        # ---------- Step 2 Geolocation ----------

        geolocation_message = TextMessage(
            content=self.geolocation_prompt.format(visual_output),
            source="user",
        )

        geolocation_response = await self.geolocation_agent.on_messages([geolocation_message], 
                                            cancellation_token = self.cancellation_token)

        final_output = geolocation_response.chat_message.content
        geolocation_usage = self._get_usage(geolocation_response.chat_message.models_usage)

        return ((visual_output, final_output), 
                (visual_usage, geolocation_usage))

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
            self.visual_analyst.on_reset(self.cancellation_token),
            self.geolocation_agent.on_reset(self.cancellation_token)
        )

    async def predict(self, image_path: str, prompt: str):
        # Note: prompt not needed
        outputs = await self.run(image_path)
        if self.save_responses:
            self.save_response(image_path, outputs)
        final_response = outputs[0][1]
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
            "visual_analyst_client_info": self.visual_analyst_client_info,
            "geolocation_client_info": self.geolocation_client_info,
            "image_path": image_path,
            "visual_analyst_output": outputs[0][0],
            "geolocation_output (final_output)": outputs[0][1],
            "usage": {
                "visual_usage": outputs[1][0],
                "geolocation_usage": outputs[1][1]
            },
            "timestamp": timestamp
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return save_path

if __name__ == "__main__":
    pass
