import torch
import torch.nn as nn
import os
import re
import time

from src.utils import nethook
from src.utils import causal_trace
from src.utils.modeltokenizer import *

if __name__ == "__main__":
    mt = ModelAndTokenizer("gpt2-xl", low_cpu_mem_usage=True, torch_dtype=torch.float16)
    print(mt)
    
    # code can be found in ./notebooks/final.ipynb
