# type declarations
from typing import Callable

from jax.numpy import DeviceArray

Model = Callable[[DeviceArray], DeviceArray]
Guide = Callable[[DeviceArray], None]
