import jax
import jax.numpy as jnp


def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty=None):

    logit_penalties = jnp.ones(logits.shape)

    if repetition_penalty is not None:
        prev_input_ids = jnp.unique(input_ids) # [[123,234,123,...]]
        logit_penalized = logits[:, prev_input_ids]
        logit_penalties = jnp.zeros(logit_penalized.shape)
        # if previous logit score is < 0 then multiply repetition penalty else divide
        logit_penalties[:, logit_penalized < 0] = repetition_penalty
        logit_penalties[:, logit_penalized > 0] = 1 / repetition_penalty

    return logit_penalties


def repetition_penalty(input_ids, logits, repetition_penalty=None):

    # Repition Penalty
    print('CTX:')
    print(input_ids)
    print('Logits:')
    print(logits)
    print('Options:')
    print("repetition_penalty:", repetition_penalty)

    penalties = _create_next_token_logits_penalties(input_ids, logits, repetition_penalty)
    logits = jnp.multiply(logits, penalties)
    return logits
