from .generator import LSTMGenerator

class Service:
    def __init__(self, config):
        self.generator = LSTMGenerator(config)

    async def generate_joke(self, prompt, maxlen=128, temperature=0.5) -> str:
        joke = self.generator.generate_txt(prompt=prompt, maxlen=maxlen, temperature=temperature)
        return joke