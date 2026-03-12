from autogen_agentchat.agents import AssistantAgent


def create_vision_agent(model_client):

    system_prompt = """
You are a visual perception specialist.

Your expertise is extracting structured visual observations from images.

You are able to carefully observe visual details and describe them in structured form so that other agents can reason about them.
"""
    model_description = "Identifies and summarizes visual evidence in the image that may indicate geographic location."
    
    agent = AssistantAgent(
        name="VisionAgent",
        model_client=model_client,
        description=model_description,
        system_message=system_prompt
    )

    return agent