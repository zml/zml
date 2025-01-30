import jax
import jax.numpy as jnp
from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')

jax.config.update("jax_traceback_in_locations_limit", -1)

@jax.jit
def benchmark(a, b):
    return jnp.dot(a, b)

# Main function
def main():
    size = 8192
    dtype = jnp.bfloat16

    rng_key = jax.random.PRNGKey(0)
    a = jax.random.uniform(rng_key, shape=(size, size), dtype=dtype)
    b = jax.random.uniform(rng_key, shape=(size, size), dtype=dtype)

    print("Compiling the benchmark function...")
    compiled_benchmark = jax.jit(benchmark)

    print("Running warm-up execution...")
    compiled_benchmark(a, b)

    print("Running benchmark...")
    
    # with jax.profiler.trace("/home/zml/zml-hugo/jax-trace", create_perfetto_link=True):
    libcudart.cudaProfilerStart()
    result = compiled_benchmark(a, b)
    result2 = compiled_benchmark(result, b)
    jax.device_get(result)
    jax.device_get(result2)
    libcudart.cudaProfilerStop()

if __name__ == "__main__":
    main()

