from absl import app, flags

import os
import sys
import typing as tp
from tqdm.auto import tqdm
import numpy as np
import json
from glob import glob
from collections import defaultdict

# Torch
import torch
import torch.utils.data as data


FLAGS = flags.FLAGS

def main(_):
    pass


if __name__ == "__main__":
    app.run(main)