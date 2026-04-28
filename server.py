"""Nahual server for cytoself.

Loads a CytoselfFull model from `cytoself.trainer.autoencoder.cytoselffull`
(the 2-stage encoder/decoder VQ-VAE used in the OpenCell paper) and runs it on
incoming `NCZYX` numpy tensors. The Z dimension is dropped before forward.

The model is configured for the canonical OpenCell setup: 2-channel
(protein + nucleus) 100x100 inputs, two VQ embedding layers, and the second
quantized vector (`vqvec2`) is returned as the embedding.

Run with:
    nix run . -- ipc:///tmp/cytoself.ipc
or:
    python server.py ipc:///tmp/cytoself.ipc
"""

import os
import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

# Make local cytoself package importable when run via `nix run`.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cytoself.trainer.autoencoder.cytoselffull import CytoselfFull  # noqa: E402

address = sys.argv[1]


def setup(
    in_channels: int = 2,
    image_size: int = 100,
    num_class: int = 1311,
    output_layer: str = "vqvec2",
    weights: str | None = None,
    device: int | None = None,
    expected_tile_size: int = 1,
) -> tuple[Callable, dict]:
    """Build a CytoselfFull model and (optionally) load a checkpoint.

    Parameters
    ----------
    in_channels : int
        Number of input channels (protein + nucleus → 2 in OpenCell).
    image_size : int
        Spatial size of the (square) input tile (100 in OpenCell).
    num_class : int
        Size of the protein-classification head. Defaults to 1311
        (matches the OpenCell pretrained checkpoint). The exact value only
        affects the FC layers; embeddings (`vqvec*`) are independent of it.
    output_layer : str
        Layer to expose as the embedding. One of ``vqvec1``, ``vqvec2``,
        ``vqind1``, ``vqind2``, ``vqindhist1``, ``vqindhist2``,
        ``encoder1``, ``encoder2``. Default ``vqvec2``.
    weights : str | None
        Path to a ``.pt``/``.pth`` checkpoint. If None, random init —
        useful for smoke tests.
    device : int | None
        CUDA device index. None → cuda:0 if available, else cpu.
    expected_tile_size : int
        Divisibility constraint for the spatial dims; cytoself uses fixed
        100x100 inputs so we set this to 1 (any size accepted) and instead
        rely on the model to error if inputs are wrong.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
    else:
        torch_device = torch.device("cpu")

    model_args = {
        "input_shape": (in_channels, image_size, image_size),
        "emb_shapes": ((25, 25), (4, 4)),
        "output_shape": (in_channels, image_size, image_size),
        "fc_output_idx": [2],
        "vq_args": {"num_embeddings": 512, "embedding_dim": 64},
        "num_class": num_class,
        "fc_input_type": "vqvec",
    }
    model = CytoselfFull(**model_args)

    if weights is not None and os.path.exists(weights):
        state_dict = torch.load(weights, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Strip common prefixes.
        state_dict = {
            k.replace("module.", "").replace("model.", ""): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        load_info = {"missing": len(missing), "unexpected": len(unexpected)}
    else:
        load_info = {"missing": 0, "unexpected": 0, "weights": "random"}

    model.to(torch_device).eval()

    info = {
        "device": str(torch_device),
        "in_channels": in_channels,
        "image_size": image_size,
        "num_class": num_class,
        "output_layer": output_layer,
        "load": load_info,
    }
    processor = partial(
        process,
        model=model,
        device=torch_device,
        expected_tile_size=expected_tile_size,
        expected_channels=in_channels,
        output_layer=output_layer,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    model,
    device: torch.device,
    expected_tile_size: int,
    expected_channels: int,
    output_layer: str,
) -> torch.Tensor:
    """Forward an NCZYX numpy array through cytoself, returning an embedding tensor."""
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )
    _, _, _, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    pixels = pad_channel_dim(pixels, expected_channels)
    torch_tensor = torch.from_numpy(pixels.copy()).float().to(device)

    with torch.no_grad():
        feats = model(torch_tensor, output_layer=output_layer)
    return feats


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"cytoself server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
