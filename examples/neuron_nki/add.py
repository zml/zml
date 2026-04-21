import nki.isa as nisa
import nki.language as nl


def add_one(x_tensor):
    out_tensor = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)
    x = nl.load(x_tensor)
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=y, data=x, op0=nl.add, operand0=1.0)
    nl.store(out_tensor, value=y)

    return out_tensor
