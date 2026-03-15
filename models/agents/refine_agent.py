from autogen_agentchat.agents import AssistantAgent


def create_refine_agent(model_client):

    system_prompt = """
You are a professional geographer specializing in global street-level imagery and visual geolocation.

You are participating in a geolocation challenge. 

Your task is to determine the most likely geographic location of an image using visual evidence extracted from the image.

Guidelines:

- Base all reasoning and conclusions strictly on observable visual evidence.
- Avoid unsupported assumptions.
"""
    model_description = "An agent that iteratively improves its geolocation reasoning through self-critique and refinement."
    
    agent = AssistantAgent(
        name="RefineAgent",
        model_client=model_client,
        description=model_description,
        system_message=system_prompt
    )

    return agent