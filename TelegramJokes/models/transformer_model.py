import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class TransformerModule():
    def __init__(self, config, tokenizer):
        '''
        Init method
        '''
        self.config = config
        self.tokenizer = tokenizer

        self.model_parameters = self.config['model']['model_params']
        self.train_parameters = self.config['train_params']
        self.device = self.config['env']['device']
        self.__init_model()

        self._criterion = nn.CrossEntropyLoss(ignore_index=0)

        lr = float(self.train_parameters['lr'])
        weight_decay = float(self.train_parameters['weight_decay'])
        gamma = float(self.train_parameters['gamma'])
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=gamma)
        self.scheduler_step = self.train_parameters['scheduler_step']

        self.num_epochs = self.train_parameters['num_epochs']
        self.clip_grad = self.train_parameters['gradient_clip']
        self.loss_logging_step = self.train_parameters['loss_logging_step']

    def __init_model(self):
        '''
        Initialize model with params from config
        '''
        self._model = TransformerModel(
            **self.model_parameters
        ).to(self.device)

    def fit(self, train_dataloader, val_dataloader=None):
        '''
        Fit model
        '''
        best_loss = np.inf
        for self.epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_dataloader)
            if self.epoch%self.scheduler_step==0:
                self._scheduler.step()
                logger.info(self.generate("Заходит улитка в бар,"))
            logger.info(f"{self.epoch+1}/{self.num_epochs}: Train loss = {train_loss:.4f}, lr = {self._scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if(train_loss < best_loss):
                best_loss = train_loss

                directory = self.config['paths']['models_checkpoints']
                os.makedirs(directory, exist_ok=True)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self._model.state_dict(),
                        'loss': best_loss,
                        'model_params': self.model_parameters,
                        'training_params': self.train_parameters,
                    }, 
                    os.path.join(directory, f"{self.config['model']['name']}.pt")
                    )
    
    def train_epoch(self, train_dataloader):
        train_loss_list = []
        self._model.train()

        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {self.epoch+1}/{self.num_epochs}', postfix={'loss': '?'}) 
        for i, batch in enumerate(train_dataloader):
            x = batch[0][:,:-1].to(self.device)
            y = batch[0][:,1:].to(self.device)

            out = self._model(x)

            self._optimizer.zero_grad()
            loss = self._criterion(out.reshape(-1, out.shape[-1]), y.reshape(-1))
            train_loss_list.append(loss.item())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clip_grad)
            self._optimizer.step()

            if (i+1)%self.loss_logging_step==0:
                logger.info(f"Loss batch {i+1}: {np.mean(train_loss_list[-self.loss_logging_step:])}")

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(1)
        pbar.close()

        return np.mean(train_loss_list)
    
    def generate(self, input_text):
        input_tokens = self.tokenizer.encode(input_text)

        input_tokens = torch.tensor(input_tokens[:-1], dtype=torch.long).unsqueeze(0).to(self.device)

        temperature = 0.5
        for i in range(100):
            self._model.eval()
            model_out = self._model(input_tokens).squeeze(0)
            model_p = torch.softmax(model_out/temperature, 1)
            sample_tokens = torch.multinomial(model_p[-1], 1)
            #next_token = model(input_tokens).argmax(-1).squeeze(0)[-1]
            input_tokens = torch.concat((input_tokens, sample_tokens.unsqueeze(0)), 1)
            if input_tokens[0,-1]==3:
                break

        output_tokens = input_tokens.squeeze(0).cpu()
        output_text = self.tokenizer.decode(list(output_tokens))
        return output_text

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout, model_max_length, **kwargs):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size
        )
        # self.pe = nn.Embedding(
        #     num_embeddings=self.model_max_length,
        #     embedding_dim=self.hidden_size
        # )  
        #nn.TransformerEncoderLayer(
        encoder_layer = TransformerEncoderLayerRoPE(
            d_model=self.hidden_size, 
            nhead=self.num_heads,
            dim_feedforward=2*self.hidden_size,
            dropout=self.dropout,
            batch_first=True
            )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.head = nn.Linear(hidden_size, self.vocab_size)
    def forward(self, x):
        B, T = x.shape
        #positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        emb = self.embeddings(x) # + self.pe(positions)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        out = self.transformer_decoder(emb, mask=mask)
        out = self.head(out)
        return out

# =========================
# RoPE (Rotary Embeddings)
# =========================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        return (
            self.cos[:seq_len, :].to(x.device),
            self.sin[:seq_len, :].to(x.device),
        )


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    # q, k: (batch, heads, seq, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,seq,dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


# =========================
# Multihead Attention + RoPE
# =========================
class MultiheadAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # (seq, batch, dim) -> (batch, seq, dim)

        B, S, D = x.shape

        qkv = self.qkv_proj(x)  # (B, S, 3D)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, Hd)

        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(x, S)
        q, k = apply_rope(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)  # (B, H, S, Hd)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out


# =========================
# Transformer Encoder Layer (RoPE)
# =========================
class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=False,
        norm_first=False,
    ):
        super().__init__()

        self.self_attn = MultiheadAttentionRoPE(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("Unsupported activation")

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, attn_mask, key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if self.norm_first:
            # Pre-norm
            src = src + self._sa_block(
                self.norm1(src), src_mask, src_key_padding_mask
            )
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-norm (как в torch)
            src = self.norm1(
                src + self._sa_block(src, src_mask, src_key_padding_mask)
            )
            src = self.norm2(src + self._ff_block(src))

        return src


# =========================
# Full Encoder (stack)
# =========================
# class TransformerEncoderRoPE(nn.Module):
#     def __init__(self, encoder_layer, num_layers):
#         super().__init__()
#         self.layers = nn.ModuleList(
#             [encoder_layer for _ in range(num_layers)]
#         )

#     def forward(self, src, mask=None, src_key_padding_mask=None):
#         output = src
#         for layer in self.layers:
#             output = layer(output, mask, src_key_padding_mask)
#         return output
