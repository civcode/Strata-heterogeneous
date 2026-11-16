"""Translate native first-layer boundaries into backend-local placement.

The native worker-plan parser and stage admission remain authoritative.
"""

import re


def layer_plan(devices, split, layers=48):
    """Resolve explicit --layer-split boundaries; auto is not qualified."""
    if type(layers) is not int or layers < 2:
        raise ValueError("invalid mixed model layer count")
    names = devices.split(",")
    first = split.split(",")
    if len(names) < 2 or len(first) != len(names) - 1:
        raise ValueError("--layer-split requires one first layer per later device")
    identities = []
    for name in names:
        if not re.fullmatch(r"(?:cuda|hip):[0-9]+", name):
            raise ValueError("--hetero-devices requires cuda:N or hip:N ordinals")
        backend, ordinal = name.split(":")
        identities.append(f"{backend}:{int(ordinal)}")
    if len(set(identities)) != len(identities):
        raise ValueError("--hetero-devices contains a duplicate device")
    if any(not re.fullmatch(r"[0-9]+", item) for item in first):
        raise ValueError("--layer-split requires explicit integer boundaries")
    bounds = [0, *map(int, first), layers]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("--layer-split boundaries must increase within the model")
    return ",".join(f"{name}:{a}-{b}" for name, a, b in zip(identities, bounds, bounds[1:]))
