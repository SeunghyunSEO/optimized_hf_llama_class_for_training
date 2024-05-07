import random
import numpy as np

import torch
import torch.nn as nn

from src.modules.fused_cross_entropy import FusedProjectionPlusCrossEntropyLoss

from pdb import set_trace as Tra

def _reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _reset_grad(layer):
    for p in layer.parameters():
        p.grad = None

def get_dummy_hidden_and_labels(seed, batch_size, seq_len, hidden_dim, vocab_size, device, dtype, ignore_index=-100):
    _reset_seeds(seed)
    hidden = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    hidden.requires_grad_(True)
    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len,), device=device)
    labels[:, -5:] = ignore_index
    return hidden, labels

seed = 1234
batch_size = 2
seq_len = 155
hidden_dim = 256
vocab_size = 128

# logit_upcast, autocast = False, False
logit_upcast, autocast = True, False
# logit_upcast, autocast = False, True
# logit_upcast, autocast = True, True

ignore_index = -100
device = "cuda:0"
dtype = torch.bfloat16
reduction = "mean"

_reset_seeds(seed)
proj_layer = nn.Linear(hidden_dim, vocab_size, bias=False).to(device, dtype=dtype)
vanilla_ce_loss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
fused_ce_loss = FusedProjectionPlusCrossEntropyLoss(proj_weight=proj_layer.weight, n_loop_iters=8, reduction=reduction, ignore_index=ignore_index).to(device, dtype=dtype)

hidden, labels = get_dummy_hidden_and_labels(seed, batch_size, seq_len, hidden_dim, vocab_size, device, dtype)
logits = proj_layer(hidden)
if logit_upcast:
    logits = logits.float()
logits = logits.view(-1, vocab_size)
labels = labels.view(-1)
assert logits.size(0) == labels.size(0), f"{logits.size(0)} != {labels.size(0)}"
with torch.cuda.amp.autocast(enabled=autocast, dtype=dtype):
    loss = vanilla_ce_loss(logits, labels)
    loss.mean().backward()
    input_grad = hidden.grad
    weight_grad = proj_layer.weight.grad.clone()

_reset_grad(proj_layer)
hidden, labels = get_dummy_hidden_and_labels(seed, batch_size, seq_len, hidden_dim, vocab_size, device, dtype)
hidden_ = hidden.view(-1, hidden_dim)
labels = labels.view(-1)
assert hidden_.size(0) == labels.size(0), f"{hidden_.size(0)} != {labels.size(0)}"
with torch.cuda.amp.autocast(enabled=autocast, dtype=dtype):
    loss_ = fused_ce_loss(hidden_, labels)
    loss_.mean().backward()
    input_grad_ = hidden.grad
    weight_grad_ = proj_layer.weight.grad.clone()

# atol, rtol = 1e-08, 1e-05 
atol, rtol = 1e-2, 1e-2

print (f'''
loss         : {loss}
loss_        : {loss_}
torch.allclose(loss.float(), loss_.float(), rtol=rtol, atol=atol): {torch.allclose(loss.float(), loss_.float(), rtol=rtol, atol=atol)}

input_grad  : {input_grad}
input_grad_ : {input_grad_}
torch.allclose(input_grad, input_grad_, rtol=rtol, atol=atol): {torch.allclose(input_grad, input_grad_, rtol=rtol, atol=atol)}

weight_grad  : {weight_grad}
weight_grad_ : {weight_grad_}
torch.allclose(weight_grad, weight_grad_, rtol=rtol, atol=atol): {torch.allclose(weight_grad, weight_grad_, rtol=rtol, atol=atol)}
''')