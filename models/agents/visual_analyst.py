from autogen_agentchat.agents import AssistantAgent


def create_visual_analyst(model_client):

    system_prompt = """
You are a visual perception expert specializing in global street-level imagery.

Your expertise is extracting detailed visual observations relevant to geolocation from a street view image.
"""
    model_description = "Identifies and analyzes visual clues in the image that may indicate geographic location."
    
    agent = AssistantAgent(
        name="VisualAnalystAgent",
        model_client=model_client,
        description=model_description,
        system_message=system_prompt
    )

    return agent