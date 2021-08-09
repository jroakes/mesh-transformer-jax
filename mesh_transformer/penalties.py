from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

# Source: https://github.com/VE-FORBRYDERNE/gpt-j-6b-filter-test/blob/da7bdfafd98096f70bd4d298f54e5aabfa00fa9e/infer.ipynb?utm_source=pocket_mylist
def apply_penalty(logits, tokens, repetition_penalty):

    shift = jnp.reshape(jnp.repeat(jnp.arange(tokens.shape[0]) * logits.shape[1], tokens.shape[1]), tokens.shape)
    penalty_logits = jnp.take(logits, tokens + shift)
    penalty_logits = jnp.where(penalty_logits > 0, penalty_logits/repetition_penalty, penalty_logits*repetition_penalty)
    return logits.at[(jnp.repeat(jnp.arange(penalty_logits.shape[0]), penalty_logits.shape[1]), tokens.flatten())].set(penalty_logits.flatten())


def repetition_penalty(logits, tokens, options):

    repetition_penalty = options.get('repetition_penalty', None)

    if repetition_penalty is not None:
        logits = apply_penalty(logits, tokens, repetition_penalty)

    return logits
