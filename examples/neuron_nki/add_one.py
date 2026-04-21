try:
    import nki.language as nl  # type: ignore
except ImportError:
    import neuronxcc.nki.language as nl  # type: ignore

try:
    import nki.isa as nisa  # type: ignore
except ImportError:
    import neuronxcc.nki.isa as nisa  # type: ignore


def add_one(x_tensor):
    assert x_tensor.shape[0] <= nl.tile_size.pmax

    out_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)
    x = nl.load(x_tensor)
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=y, data=x, op0=nl.add, operand0=1.0)
    nl.store(out_tensor, value=y)

    return out_tensor
