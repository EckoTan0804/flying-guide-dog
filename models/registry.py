# Motivated by https://github.com/Julienbeaulieu/kaggle-computer-vision-competition/blob/7bc6bcb8b85d81ff1544040c403e356c0a3c8060/src/tools/registry.py#L9

from typing import Callable, Dict, Optional
import torch.nn as nn


def _register_generic(module_dict: Dict[str, nn.Module], module_name: str, module: nn.Module) -> None:
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class for managing registering modules. It extends a dictionary
    and provides a register functions.
    Eg. creating a registry:
        some_registry = Registry({"default": default_module})
        
    There're two ways of registering new modules:
    1): Normal way: just call register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
        
    2): Use as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name: str, module: nn.Module = None) -> Optional[Callable]:
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn
