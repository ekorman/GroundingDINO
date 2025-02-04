from typing import Dict
import torch
from transformers import AutoTokenizer


def get_phrases_from_posmap(
    posmap: torch.BoolTensor,
    tokenized: Dict,
    tokenizer: AutoTokenizer,
    left_idx: int = 0,
    right_idx: int = 255,
):
    assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
    if posmap.dim() == 1:
        posmap[0 : left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")
