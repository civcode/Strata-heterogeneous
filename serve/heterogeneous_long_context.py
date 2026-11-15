"""Shared CPU policy for the opt-in resident Q8 long-context path."""
MAX_LONG_CONTEXT = 262144
MIN_LONG_CONTEXT = 2048
LONG_CONTEXT_ALIGNMENT = 256
LONG_CONTEXT_RESERVE = 1 << 30
LONG_COMMAND_BYTES = 3 * 1024 * 1024
LONG_PROMPT_BYTES = 4 * 1024 * 1024
LONG_PREFILL_TIMEOUT_SECONDS = 1800


def valid_cells(cells):
    return (type(cells) is int and MIN_LONG_CONTEXT <= cells <= MAX_LONG_CONTEXT
            and cells % LONG_CONTEXT_ALIGNMENT == 0)


def require_mode(extended, workload, kv, grouped, bulk, lookup):
    if type(extended) is not bool:
        raise ValueError("long-context must be boolean")
    if extended and (workload is not True or kv != "int8" or grouped is not True or
                     type(bulk) is not int or not 1 <= bulk <= 2048 or lookup is not False):
        raise ValueError("long-context requires workload mode, grouped native HIP, INT8 KV, native prefill and no lookup")


def validate_capacity(cells, requested):
    if not valid_cells(requested) or type(cells) is not int or cells != requested:
        raise ValueError("long-context requires an exact aligned worker capacity in 2048..262144")


def command_cap(context, extended):
    if type(extended) is not bool:
        raise ValueError("long-context must be boolean")
    if extended:
        if not valid_cells(context):
            raise ValueError("invalid long-context command capacity")
        return LONG_COMMAND_BYTES
    if context > 32768:
        raise ValueError("context above 32768 requires long-context mode")
    return 384 * 1024 if context > 8192 else 65536
