import inspect
from tunix.models.qwen2 import params as qp, model as qm
import jax
import jax.numpy as jnp
import numpy as np

config = qm.ModelConfig.qwen2p5_3b()
mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape((len(jax.devices()), 1)), ('fsdp', 'tp'))
model = qp.create_model_from_safe_tensors('', config, mesh, dtype=jnp.bfloat16, init_only=True)
print(inspect.signature(model.get_model_input))
