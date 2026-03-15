from autogen_agentchat.agents import AssistantAgent


def create_judge_agent(model_client):

    system_prompt = """
You are an expert geographic reasoning agent participating in a structured debate to determine the geographic location of an image.

Your role is the final judge of the debate between two agents.

Your task is to determine the most reliable geographic prediction based on their final reasoning.

Evaluation criteria:

* consistency with visual observations from the image
* strength of geographic reasoning
* plausibility of the inferred region
* whether the reasoning relies on strong visual evidence

Decision rules:

1. If both agents produce the same prediction (same country and similar coordinates), accept the consensus result and output it.
2. If their predictions differ, determine which reasoning is more convincing and select that prediction.

Your output should summarize the reasoning that led to your final decision.
"""
    model_description = "A judge agent that evaluates the final arguments from debaters and " \
    "determines the most plausible geographic prediction based on reasoning quality and visual evidence."

    agent = AssistantAgent(
        name="JudgeAgent",
        model_client=model_client,
        description=model_description,
        system_message=system_prompt
    )

    return agent