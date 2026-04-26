import jax
import jax.numpy as jnp
import numpy as np
import tunix
from tunix.models.qwen2 import model as qm, params as qp

config = qm.ModelConfig.qwen2p5_3b()
mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape((len(jax.devices()), 1)), ('fsdp', 'tp'))
model = qp.create_model_from_safe_tensors('', config, mesh, dtype=jnp.bfloat16)
inputs = model.get_model_input()

print('KEYS:', inputs.keys())
print('SHAPES:', {k: v.shape for k, v in inputs.items()})
