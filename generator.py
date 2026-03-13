import torch

class JokeGenerator:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(self, prompt="", max_len=100):

        tokens = self.tokenizer.encode(prompt)

        for _ in range(max_len):

            x = torch.tensor(tokens).unsqueeze(0).to(self.device)

            logits = self.model(x)
            next_token = logits[0, -1].argmax().item()

            tokens.append(next_token)

            if next_token == self.tokenizer.eos_id:
                break

        return self.tokenizer.decode(tokens)