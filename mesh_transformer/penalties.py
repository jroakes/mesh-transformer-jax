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


def apply_repetition_penalty(sequences,
                             logits,
                             i,
                             repetition_penalty,
                             repetition_window,
                             repetition_penalty_normalize):
    # https://github.com/google-research/google-research/blob/34444253e9f57cd03364bc4e50057a5abe9bcf17/protein_lm/sampling.py

    max_i = i  # We are currently generating a token for position i + 1.
    min_i = i - repetition_window + 1
    batch_size, vocab_size = logits.shape
    positions = jnp.arange(sequences.shape[1])
    positions = jnp.tile(positions[jnp.newaxis, :, jnp.newaxis],
                       [batch_size, 1, vocab_size])
    sequences_onehot = jnp.eye(vocab_size)[sequences]
    sequences_onehot = jnp.where((positions >= min_i) * (positions <= max_i),
                               sequences_onehot,
                               jnp.zeros_like(sequences_onehot))
    # Compute the indicator that a token appeared at least once in the
    # repetition window. Output shape: (batch_size, vocab_size).
    indicator = jnp.max(sequences_onehot, axis=1)
    # Compute a penalty tensor. The logits are divided by the penalty tensor.
    penalty_tensor = jnp.where(indicator,
                             jnp.ones_like(logits) * repetition_penalty,
                             jnp.ones_like(logits))

    if repetition_penalty_normalize:
        logits = jax.nn.log_softmax(logits)
    # Dividing a negative logit by the penalty tensor actually increases the
    # resulting probability. Take the inverse for negative logits.
    penalty_tensor = jnp.where(logits > 0,
                             penalty_tensor,
                             1 / penalty_tensor)

    logits = logits / penalty_tensor
    return logits


def repetition_penalty(input_ids, i, logits, options):

    repetition_penalty = options.get('repetition_penalty', 1)
    repetition_window = options.get('repetition_window', 4)
    repetition_penalty_normalize = options.get('repetition_penalty_normalize', False)


    #if repetition_penalty != 1: TODO: Need to figure out how to compare bool with jax parameters
    logits = apply_repetition_penalty(  input_ids,
                                         logits,
                                         i,
                                         repetition_penalty,
                                         repetition_window,
                                         repetition_penalty_normalize)

        #penalties = _create_next_token_logits_penalties(input_ids, logits, repetition_penalty)
        #logits = jnp.multiply(logits, penalties)
    return logits
