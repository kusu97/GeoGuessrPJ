class PromptManager():
    """
    A class managing all the prompts.

    geobench_prompt: 
        taken from geobench: https://github.com/ccmdi/geobench
    """

    def __init__(self):
        self.prompt_path = {"geobench": "./prompts/geobench_prompt.txt"}
    
    def get_prompt(self, prompt_name):
        try:
            path = self.prompt_path[prompt_name]
        except KeyError:
            raise KeyError(f"Prompt '{prompt_name}' not found.")

        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt


if __name__ == '__main__':
    pass