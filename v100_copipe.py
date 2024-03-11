#!/usr/bin/env python3

import os
import torch
from torch.nn import Embedding, TransformerEncoderLayer, Linear, LogSoftmax, NLLLoss
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.ops.adam import DeepSpeedCPUAdam

training_config = {
    "num_stages": 4,
    "seq_len": 1536,
    "micro_batch_size": 2,
    "steps": 30,
}
training_config["batch_size"] = training_config["micro_batch_size"] * (
    2 * training_config["num_stages"])

model_config = {
    "vocab_size": 50272,
    "pad_token_id": 1,
    "num_layers": 18 * training_config["num_stages"],
}

layer_config = {
    "d_model": 2560,
    "nhead": 32,
    "dim_feedforward": 10240,
    "activation": "relu",
    "batch_first": True,
    "norm_first": True,
}


class RandomDataset(torch.utils.data.Dataset):

    def __init__(self, vocab_size, seq_len, size):
        self.size = size
        self.samples = [
            torch.randint(vocab_size, (seq_len, )) for _ in range(size)
        ]
        self.targets = [
            torch.randint(vocab_size, (seq_len, )) for _ in range(size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


def loss_func(outputs, targets):
    criterion = NLLLoss()
    return criterion(outputs.transpose(1, 2), targets)


def train_pipeline(local_rank):
    steps = training_config["steps"]
    layers = [
        LayerSpec(
            Embedding,
            model_config["vocab_size"],
            layer_config["d_model"],
            padding_idx=model_config["pad_token_id"],
        )
    ]
    layers.extend([
        LayerSpec(
            TransformerEncoderLayer,
            **layer_config,
        ) for _ in range(model_config["num_layers"])
    ])
    layers.append(
        LayerSpec(
            Linear,
            layer_config["d_model"],
            model_config["vocab_size"],
            bias=False,
        ))
    layers.append(LayerSpec(LogSoftmax, dim=-1))

    model = PipelineModule(
        layers=layers,
        num_stages=training_config["num_stages"],
        loss_fn=loss_func,
        partition_method="type:TransformerEncoderLayer",
    )
    train_set = RandomDataset(
        model_config["vocab_size"],
        training_config["seq_len"],
        training_config["batch_size"] * training_config["steps"],
    )
    ds_config = {
        "train_batch_size": training_config["batch_size"],
        "train_micro_batch_size_per_gpu": training_config["micro_batch_size"],
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8,
        },
        "pipeline": {
            "CoPipe_ratios": [{
                0: 0.18,
                1: 0.25,
                2: 0.25,
                3: 0.32,
            }, {
                -2: 0.15,
                -1: 0.30,
                1: 0.15,
                2: 0.15,
                3: 0.25,
            }, {
                -2: 0.18,
                -1: 0.57,
                2: 0.25,
            }, {
                -2: 0.20,
                -1: 0.80,
            }]
        },
        "steps_per_print": steps if steps <= 10 else 10,
    }

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=DeepSpeedCPUAdam,
        training_data=train_set,
        config=ds_config,
    )

    # prof = profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=schedule(wait=1, warmup=2, active=2),
    #     on_trace_ready=tensorboard_trace_handler(
    #         f"./tensorboard_logs/pipeline_baseline"),
    #     with_stack=True,
    # )
    # prof.start()
    for _ in range(steps):
        engine.train_batch()
    #     prof.step()
    # prof.stop()

    allocated = [
        f"{(torch.cuda.max_memory_allocated(rank) / float(2**30)):.3f}"
        for rank in range(training_config["num_stages"])
    ]
    reserved = [
        f"{(torch.cuda.max_memory_reserved(rank) / float(2**30)):.3f}"
        for rank in range(training_config["num_stages"])
    ]
    print(
        f"[Rank {local_rank}] | Max cuda memory allocated {allocated} GB | reserved {reserved} GB"
    )


if __name__ == "__main__":
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    train_pipeline(local_rank)
