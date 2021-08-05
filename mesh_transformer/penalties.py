import jax
import jax.numpy as jnp
import numpy as np


def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty):

    print('input_ids', input_ids)
    print('logits', logits)
    print('Repetiton Penalty', repetition_penalty)
    
    if type(input_ids) != jax.interpreters.partial_eval.DynamicJaxprTracer:
        prev_input_ids = jnp.unique(input_ids) # [[123,234,123,...]]
        logit_penalized = logits[:, prev_input_ids]
        logit_penalties = jnp.zeros(logit_penalized.shape)
        # if previous logit score is < 0 then multiply repetition penalty else divide
        logit_penalties[:, logit_penalized < 0] = repetition_penalty
        logit_penalties[:, logit_penalized > 0] = 1 / repetition_penalty

        logits - jnp.multiply(logits, logit_penalties)

    return logits




def repetition_penalty(input_ids, i, logits, options):

    repetition_penalty = options.get('repetition_penalty', 1)

    penalties = _create_next_token_logits_penalties(input_ids, logits, repetition_penalty)
    return logits
