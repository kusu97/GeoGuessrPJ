class PromptManager():
    """
    A class managing all the prompts.

    geobench_prompt: 
        (taken from geobench: https://github.com/ccmdi/geobench)
        A structured geolocation prompt with chain-of-thought reasoning and enforced output schema. 
        Designed for consistent evaluation and automatic parsing, 
        while serving as a baseline to study how reasoning instructions affect model behavior.

    Besides, we design four new prompts to systematically study how reasoning 
    intensity and expert role prompting influence MLLM geolocation performance.

    All these four prompts share the same output format constraints to ensure 
    fair automatic evaluation. The prompts differ only in reasoning requirements
    and role specification, allowing controlled comparison.

    1. explicit_CoT: High reasoning intensity
        Explicitly instructs the model to analyze multiple categories of visual
        evidence (signage, vegetation, architecture, terrain, climate, etc.) and 
        perform step-by-step reasoning. This prompt enforces strong chain-of-thought 
        and evidence-driven inference.

    2. light_reasoning: Weak reasoning intensity
        Encourages brief reasoning before prediction, but does not explicitly
        enumerate visual features. This tests whether light reasoning improves
        performance without imposing heavy cognitive structure.

    3. direct_prediction: No reasoning
        Requires direct prediction with no explanation. This isolates the effect
        of chain-of-thought by measuring performance when reasoning is suppressed.

    4. expert_persona: Expert persona
        Identical to Prompt "light_reasoning" except for adding a professional 
        geographer role. This isolates the effect of expert role prompting while 
        keeping reasoning instructions constant.

    This design forms a controlled prompt study where only one dimension (reasoning 
    strength or role specification) changes at a time.
    """

    def __init__(self):
        self.prompt_path = {"geobench": "./prompts/geobench_prompt.txt",
                            "explicit_CoT": "./prompts/explicit_CoT.txt",
                            "light_reasoning": "./prompts/light_reasoning.txt",
                            "direct_prediction": "./prompts/direct_prediction.txt",
                            "expert_persona": "./prompts/expert_persona.txt"}
    
    def get_prompt(self, prompt_name):
        try:
            path = self.prompt_path[prompt_name]
        except KeyError:
            raise KeyError(f'Prompt "{prompt_name}" not found.')

        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt


if __name__ == '__main__':
    example = PromptManager()
    for prompt_name in example.prompt_path.keys():
        prompt = example.get_prompt(prompt_name)
        print(prompt, end="\n\n")