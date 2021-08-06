from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

@partial(jit, static_argnums=(2, 3))
def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty, repetition_window):

    prev_input_ids = input_ids[:, -repetition_window:].squeeze()

    logit_penalized = logits[:, prev_input_ids]
    logit_penalties = jnp.zeros(logit_penalized.shape)

    # if previous logit score is < 0 then multiply repetition penalty else divide
    logit_penalties = jnp.where(logit_penalized < 0, logit_penalties, repetition_penalty)
    logit_penalties = jnp.where(logit_penalized > 0, logit_penalties, 1 / repetition_penalty)
    token_penalties = jax.ops.index_update(token_penalties, jax.ops.index[:, prev_input_ids], logit_penalties)

    logits = jnp.multiply(logits, token_penalties)

    return logits




def repetition_penalty(input_ids, i, logits, options):

    repetition_penalty = options.get('repetition_penalty', 1)
    repetition_window = options.get('repetition_window', 10)

    penalties = _create_next_token_logits_penalties(input_ids, logits, repetition_penalty, repetition_window)
    return logits
