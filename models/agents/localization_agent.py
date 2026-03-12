from autogen_agentchat.agents import AssistantAgent


def create_localization_agent(model_client):

    system_prompt = """
You are a geographic localization specialist.

You estimate the most likely real-world location based on reasoning provided by other agents.

You provide a clear and concise final prediction.
"""
    model_description = "Produces the final geolocation prediction, including country and coordinates, based on geographic reasoning."

    agent = AssistantAgent(
        name="LocalizationAgent",
        model_client=model_client,
        description=model_description,
        system_message=system_prompt
    )

    return agent