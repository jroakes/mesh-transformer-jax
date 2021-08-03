import jax
import jax.numpy as jnp
import numpy as np


def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty):

    print('input_ids', input_ids)
    print('logits', logits)
    print('Repetiton Penalty', repetition_penalty)

    prev_input_ids = np.unique(input_ids) # [[123,234,123,...]]
    logit_penalized = logits[:, prev_input_ids]
    logit_penalties = jnp.zeros(logit_penalized.shape)
    # if previous logit score is < 0 then multiply repetition penalty else divide
    logit_penalties[:, logit_penalized < 0] = repetition_penalty
    logit_penalties[:, logit_penalized > 0] = 1 / repetition_penalty

    return logit_penalties



def repetition_penalty(input_ids, logits, options):

    repetition_penalty = options.get('repetition_penalty', None)

    if repetition_penalty is not None:
        penalties = _create_next_token_logits_penalties(input_ids, logits, repetition_penalty)
        logits = jnp.multiply(logits, penalties)
    return logits
