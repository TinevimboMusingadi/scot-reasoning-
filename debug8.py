import jax, jax.numpy as jnp, numpy as np
from tunix.models.qwen2 import model as qm, params as qp

try:
    config = qm.ModelConfig.qwen2p5_3b()
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape((len(jax.devices()), 1)), ('fsdp', 'tp'))
    model = qp.create_model_from_safe_tensors('', config, mesh, dtype=jnp.bfloat16)

    def print_tree(tree, path=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                print_tree(v, path + "." + k if path else k)
        elif hasattr(tree, "shape"):
            print(f"{path}: {tree.shape}")
        else:
            print(f"{path}: {type(tree)}")

    print_tree(model.params)
except Exception as e:
    print("ERROR", e)
