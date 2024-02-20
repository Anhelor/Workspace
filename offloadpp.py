#!/usr/bin/env python3

import os
import torch
from torch.nn import Embedding, TransformerEncoderLayer, Linear, LogSoftmax, NLLLoss
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

training_config = {
    "num_stages": 4,
    "seq_len": 2048,
    "micro_batch_size": 4,
    "steps": 90,
}
training_config["batch_size"] = training_config["micro_batch_size"] * (
    2 * training_config["num_stages"])

model_config = {
    "vocab_size": 30522,
    "pad_token_id": 1,
    "num_layers": 20 * training_config["num_stages"],
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
                -1: 0.95,
                0: 0.05,
            }, {
                -1: 0.95,
                1: 0.05,
            }, {
                -1: 0.95,
                2: 0.05,
            }, {
                -1: 0.95,
                3: 0.05,
            }]
        },
        "steps_per_print": steps if steps <= 15 else 15,
    }

    _, _ = (DeepSpeedCPUAdam([torch.nn.Parameter(torch.randn(1))]),
            FusedAdam([torch.nn.Parameter(torch.randn(1))]))

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=DeepSpeedCPUAdam,
        training_data=train_set,
        config=ds_config,
    )

    for _ in range(steps):
        engine.train_batch()

    allocated = [
        f"{(torch.cuda.max_memory_allocated(rank) / float(2**30)):.3f}"
        for rank in [0, 1, 2, 3]
    ]
    reserved = [
        f"{(torch.cuda.max_memory_reserved(rank) / float(2**30)):.3f}"
        for rank in [0, 1, 2, 3]
    ]
    print(
        f"[Rank {local_rank}] | Max cuda memory allocated {allocated} GB | reserved {reserved} GB"
    )


if __name__ == "__main__":
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    train_pipeline(local_rank)
