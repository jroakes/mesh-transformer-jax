import argparse
import json
import random
import time

import jax
import numpy as np
import pandas as pd
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--sample_file", type=str, default=None, help="Pandas DataFrame")
    parser.add_argument("--prompt_column", type=str, default='model_input', help="Pandas DataFrame")
    parser.add_argument("--num_samples", type=int, default=10, help="Pandas DataFrame")
    parser.add_argument("--ckpt_step", type=int, default=0, help="Pandas DataFrame")
    parser.add_argument("--temp", type=float, default=0.5, help="Prediction Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Prediction Top-P")
    parser.add_argument("--max_len", type=int, default=300, help="Prediction Max Length")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))
    sample_file = args.sample_file
    prompt_column = args.prompt_column
    num_samples = args.num_samples
    pred_temp = args.temp
    pred_top_p = args.top_p
    pred_max_len = args.max_len
    ckpt_step = args.ckpt_step

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    if ckpt_step == 0:
        ckpt_step = meta["checkpoints"][-1]

    print(f"using checkpoint {ckpt_step}")

    samples_df = pd.read_csv(sample_file)
    samples = random.choices(samples_df[prompt_column].tolist(), k=num_samples)
    df_result = pd.DataFrame()

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica

    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

        for sample in samples:

            tokens = tokenizer.encode(sample)
            print('Input Context:', sample)
            print()

            start = time.time()

            provided_ctx = len(tokens)
            pad_amount = seq - provided_ctx

            padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
            batched_tokens = np.array([padded_tokens] * total_batch)
            length = np.ones(total_batch, dtype=np.uint32) * pred_max_len

            output = network.generate(batched_tokens, length, 512, {"top_p": np.ones(total_batch) * pred_top_p,
                                                                    "temp": np.ones(total_batch) * pred_temp})

            print('Ouput generations:', len(output[1][0][:, :, 0]))
            encoded_tokens = list(output[1][0][:, :, 0][0])

            end_tokens = [198, 50256] # \n or <|endoftext|>
            stop_idx = len(encoded_tokens)
            for end_token in end_tokens:
                if end_token in encoded_tokens:
                    stop_idx = encoded_tokens.index(tokenizer.eos_token_id)
                    break

            decoded_tokens = tokenizer.decode(encoded_tokens[:stop_idx])

            print('Decoded Tokens:', decoded_tokens)
            print()

            df_result = df_result.append({'prompt': sample, 'predicted':decoded_tokens}, ignore_index=True)


            print(f"completion done in {time.time() - start:06}s")

    df_result.to_csv('result_'+sample_file, index=None)
    print('Output Saved:', 'result_'+sample_file)
