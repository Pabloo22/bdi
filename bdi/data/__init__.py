"""Prompt sources for distillation runs, saved in the parquet format verl's RLHFDataset reads."""

import os
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_chat_parquet(prompts: Sequence[str], path: Path, data_source: str) -> None:
    """Save one single-turn chat per prompt: the `prompt` column holds a list of chat messages.

    The file appears atomically, so runs started at the same time never read a partial file.
    """
    table = pa.table(
        {
            "prompt": [[{"role": "user", "content": prompt}] for prompt in prompts],
            "data_source": [data_source] * len(prompts),
            "extra_info": [{"index": i} for i in range(len(prompts))],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
