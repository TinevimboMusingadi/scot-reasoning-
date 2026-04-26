import jax
import jax.numpy as jnp
from tunix.models.qwen2 import model as qwen_lib, params as qwen_params
import numpy as np

config = qwen_lib.ModelConfig.qwen2p5_3b()
mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape((len(jax.devices()), 1)), ('fsdp', 'tp'))

model = qwen_params.create_model_from_safe_tensors('', config, mesh, dtype=jnp.bfloat16, init_only=True)
inputs = model.get_model_input()

print('MODEL INPUTS DUMMY FOR FLATTENED QWEN 2.5 3B:')
for k, v in inputs.items():
    print(f"Key: {k}, Shape: {v.shape}")
