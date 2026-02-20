import jax
import numpy as np
import jax.numpy as jnp
import plotly.express as plt
import functools
np.set_printoptions(threshold=np.inf)
# ! XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/
# ! XLA_FLAGS = --xla_force_host_platform_device_count=8
print(jax.devices())
# print(jax.lib.xla_bridge.get_backend().platform)



@functools.partial(jax.jit, static_argnames=["steps", "nu", "dt", "dx", "dy", "nx", "ny"])
def diffuse_on(u, steps, nu, dt, dx, dy, nx, ny
               ) -> jnp.ndarray:
    def diffuse(i, u) -> jnp.ndarray:
        u_view = jax.lax.dynamic_slice(
            u, start_indices=(1, 1), slice_sizes=(nx-2, ny-2))
        u_view_x = nu * dt / dx**2 * (jax.lax.dynamic_slice(u, start_indices=(1, 2), slice_sizes=(nx-2, ny-2))
                                      + jax.lax.dynamic_slice(u, start_indices=(1, 0), slice_sizes=(nx-2, ny-2))
                                      - 2 * u_view)
        u_view_y = nu * dt / dy**2 * (jax.lax.dynamic_slice(u, start_indices=(2, 1), slice_sizes=(nx-2, ny-2))
                                      + jax.lax.dynamic_slice(u, start_indices=(0, 1), slice_sizes=(nx-2, ny-2))
                                      - 2 * u_view)
        return jax.lax.dynamic_update_slice(u, update=(
            u_view + u_view_x + u_view_y), start_indices=(1, 1))
    return jax.lax.fori_loop(0, steps, diffuse, u)

# fig = plt.imshow(jax.numpy.array(u),
#                  zmin=1,
#                  zmax=2,
#                  color_continuous_scale='RdBu_r',
#                  labels={"color": "Heat"})
# fig.show()