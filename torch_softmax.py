import torch
import torch.nn as nn
import torch._dynamo
import torch._dynamo.config
import torch._inductor.config
import logging
import sys
import os

from torch.profiler import profile, record_function, ProfilerActivity

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

torch._dynamo.reset()

def timing(fn, *args):
    from time import time
    s = time()
    fn(*args)
    print("elapsed time:", (time() - s) * 1e6, "us")


def fn(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


class MyModel(nn.Module):
    def forward(self, x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        c = a + b
        return torch.softmax(c, -1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel()
opt_model = torch.compile(model)

a = torch.randn(10, device=device)
b = torch.randn(10, device=device)
c = torch.randn(2, device=device)


with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    with record_function("model_inference"):
        for _ in range(10):
            opt_model(a, b)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
prof.export_chrome_trace(os.path.basename(sys.argv[0]) + ".json")
