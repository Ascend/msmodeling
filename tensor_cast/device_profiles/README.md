# Device Profiles

Put your device profile Python files under this folder so TensorCast can automatically load them.

## How To Define A Custom Device

1. Create a new Python file in this folder, for example `my_device.py`.
2. Import `DeviceProfile` and define a profile with a unique `name`.
3. Set compute, memory, and interconnect fields that match your hardware.

Minimal example:

```python
from tensor_cast.device import DeviceProfile, CommGrid, InterconnectTopology, InterconnectType
import torch

MY_DEVICE_INTERCONNECT = CommGrid(
    grid=torch.arange(8).reshape(8),
    topologies={
        0: InterconnectTopology(
            bandwidth_bytes_ps=64 * 1e9,
            latency_s=0.2e-6,
            comm_efficiency=0.7,
            type=InterconnectType.FULL_MESH,
        ),
    },
)

MY_DEVICE = DeviceProfile(
    name="MY_DEVICE",
    vendor="MY_VENDOR",
    mma_ops={torch.float32: 100 * 1e12, torch.bfloat16: 200 * 1e12},
    gp_ops={torch.float32: 10 * 1e12, torch.bfloat16: 20 * 1e12},
    memory_size_bytes=64 * (1024**3),
    memory_bandwidth_bytes_ps=1.6 * (1024**4),
    compute_efficiency=0.7,
    memory_efficiency=0.6,
    comm_grid=MY_DEVICE_INTERCONNECT,
)
```

Once defined, your profile is available in `DeviceProfile.all_device_profiles`.
