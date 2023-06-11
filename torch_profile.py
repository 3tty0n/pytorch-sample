import torch
import torch._dynamo
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import sys
import time
import logging

torch._dynamo.config.log_level=logging.DEBUG   # Better to set TORCH_COMPILE_DEBUG?
# torch._dynamo.config.verbose=True                # TORCHDYNAMO_VERBOSE=True
torch._dynamo.config.output_code=True
# torch._dynamo.config.output_graph_code=True

# import torch._inductor.config
# torch._inductor.config.verbose_progress=True
# torch._inductor.config.debug=True
# torch._inductor.config.trace.enabled=True        # TORCH_COMPILE_DEBUG=1
# torch._inductor.config.trace.info_log=True
# torch._inductor.config.pattern_matcher=False
# torch._inductor.config.aggressive_fusion=True
# torch._inductor.config.permute_fusion=True      # TORCHINDUCTOR_PERMUTE_FUSION=1

torch._dynamo.reset()

batch_size = 64

device = torch.device("cuda")

inputs = torch.randn(batch_size, 3, 224, 224, device=device)
model = models.resnet18().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

compiled_model = torch.compile(
    model,
    backend="inductor",
    options={
        #"debug": True,
        #"trace.enabled": True,
        #"trace.debug_log": True,
        #'trace.graph_diagram': True,
        "verbose_progress": True,
    },
)


def warmup(N):
    for i in range(N):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        start = time.time()
        out = compiled_model(inputs)
        torch.cuda.synchronize()

        forward_elapsed_time = time.time() - start

        torch.cuda.synchronize()
        start = time.time()
        out.sum().backward()
        backward_elapsed_time = time.time() - start

        print(
            f"with compile {i} iter forward: {forward_elapsed_time/1000:.3e} msec., backward: {backward_elapsed_time/1000:.3e} msec."
        )
        optimizer.step()

    print("-" * 10)


def evaluate(N):
    torch.cuda.synchronize()
    all_start = time.time()
    for i in range(N):
        optimizer.zero_grad()
        out = compiled_model(inputs)
        out.sum().backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed_time = time.time() - all_start
    print(
        f"with compile total:{elapsed_time:.3e} sec. {batch_size*N/elapsed_time:.3e} imgs/sec."
    )


def test_with_profile():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            #warmup(10)
            evaluate(10)

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    prof.export_chrome_trace(sys.argv[0].rstrip(".py") + ".json")


if __name__ == "__main__":
    test_with_profile()
