import tunix.models.qwen2 as q2
print("DIR:", dir(q2))
try:
    print("Safetensors param available?", hasattr(q2, "params_safetensors"))
except Exception as e:
    pass
