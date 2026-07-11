import torch


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=100, temperature=1.0, top_k=0, device=None):
    device = device or next(model.parameters()).device
    model.to(device)
    model.eval()

    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        idx_cond = idx[:, -model.context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())
