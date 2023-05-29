import torch
import torch._dynamo as dynamo
import logging

#torch._dynamo.config.verbose=True
#torch._dynamo.config.log_level = logging.INFO

import torch._inductor.config
#torch._dynamo.config.log_level = logging.INFO
#torch._inductor.config.verbose_progress=True
#torch._inductor.config.debug=True
#torch._inductor.config.trace.enabled=True

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    #@dynamo.optimize()
    def forward(self, x):
        x = torch.permute(x, (1,0))
        x = torch.add(x, x[2,:])
        return x

model = MyModel()
#model = torch.compile(model, fullgraph=True)  // an exception occurs when this is executed.
model = torch.compile(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([[1.,2.,3.],
                  [4.,5.,6.],
                  [7.,8.,9.]], device=device)

y = model(x)
