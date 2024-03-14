import warnings
import jittor as jt

class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter('ignore')
        return self

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def assert_shape(var, ref_shape):
    if var.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {var.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(var.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, jt.Var):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                assert(jt.equal(jt.Var(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, jt.Var):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                assert(jt.equal(size, jt.Var(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')
