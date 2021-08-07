from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np


def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty, i):

    print("input_ids:", input_ids)
    print("logits:", logits)
    print("repetition_penalty:", repetition_penalty)
    print("repetition_window:", i)

    token_penalties = jnp.ones(logits.shape)

    if type(i) != jax.interpreters.partial_eval.DynamicJaxprTracer:
        prev_input_ids = jax.lax.dynamic_slice(input_ids, (0, -i), (input_ids.shape[0], i))
    else:
        prev_input_ids = jax.lax.dynamic_slice(input_ids, (0, -100), (input_ids.shape[0], 100))

    # IndexError: Array slice indices must have static start/stop/step to be used with NumPy
    # indexing syntax. To index a statically sized array at a dynamic position,try
    # lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays
    # within JIT compiled functions).

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

    logits = _create_next_token_logits_penalties(input_ids, logits, repetition_penalty, i+1)

    return logits
