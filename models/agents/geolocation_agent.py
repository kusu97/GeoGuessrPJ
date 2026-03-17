from autogen_agentchat.agents import AssistantAgent


def create_geolocation_agent(model_client):

    system_prompt = """
You are a professional geographer specializing in global geolocation.

Your expertise is inferring the most likely geographic location based on the visual observations extracted from a street view image.
"""
    model_description = "Infers the most likely geographic location based on visual observations."

    agent = AssistantAgent(
        name="GeolocationAgent",
        model_client=model_client,
        description=model_description,
        system_message=system_prompt
    )

    return agent