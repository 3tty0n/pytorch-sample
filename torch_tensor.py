import torch
import torch._dynamo
import torch._inductor
from torch.profiler import profile, record_function, ProfilerActivity

import logging
from pprint import pprint

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True


def f(x):
    return torch.sin(x) ** 2 + torch.cos(x) ** 2

torch._dynamo.reset()

compiled_f = torch.compile(f, backend='inductor',
                              options={'debug': True,
                                       'trace.enabled': True,
                                       'trace.debug_log': True,
                                       'trace.graph_diagram': True,
                                       'verbose_progress': True
                                       })


torch.manual_seed(0)

device = torch.device('cuda')
# Create a tensor on CPU
tensor = torch.rand(1000, requires_grad=True)

# Allocate a tensor on GPU
x_tensor = tensor.to(device)
y_tensor = torch.ones_like(x_tensor)

out = torch.nn.functional.mse_loss(compiled_f(x_tensor), y_tensor).backward()
