import torch

class JokeGenerator:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(self, prompt="", maxlen=128, temperature=0.5):

        tokens = self.tokenizer.encode(prompt)[:maxlen]
        generated = tokens[:-1]
        input = torch.tensor(generated, dtype=torch.long).unsqueeze(0)

        h, c = None, None
        for i in range(maxlen):
            emb = self.model.embeddings(input)
            if h is None:
                out, (h,c) = self.model.encoder(emb)
            else:
                out, (h,c) = self.model.encoder(emb, (h, c))
            logits = self.model.head(out)[:,-1] / temperature
            probs = torch.softmax(logits, -1)
            input = torch.multinomial(probs[-1], 1).unsqueeze(0)
            generated.append(input.item())
            if generated[-1]==3:
                break
        output_text = self.tokenizer.decode(generated)
        return output_text