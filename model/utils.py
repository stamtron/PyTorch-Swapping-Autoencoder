import torch
from torch.autograd import Function
from itertools import cycle
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd

from sklearn.model_selection import train_test_split

import tqdm
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss

from torch.nn.parameter import Parameter
import torchvision
from tqdm import tqdm

import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, jaccard_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
