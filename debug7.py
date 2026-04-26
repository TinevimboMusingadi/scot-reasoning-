import qwix
import inspect
print("LORA:", inspect.signature(qwix.LoraProvider))
print("LORA INIT:", inspect.signature(qwix.LoraProvider.__init__))
