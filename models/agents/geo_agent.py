from autogen_agentchat.agents import AssistantAgent


def create_geo_agent(model_client):

    system_prompt = """
You are a geographic reasoning specialist.

You analyze structured evidence and infer possible geographic regions using world knowledge such as climate, language, vegetation, architecture and infrastructure patterns.

You explain your reasoning clearly so that other agents can use it.
"""
    model_description = "Translates visual evidence into geographic hypotheses about possible regions or countries."

    agent = AssistantAgent(
        name="GeoReasoningAgent",
        model_client=model_client,
        description=model_description,
        system_message=system_prompt
    )

    return agent