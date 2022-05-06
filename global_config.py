import os
import multiprocessing
from multiprocessing import Lock

# define single
Locks = [multiprocessing.Lock()]
multiprocessing = multiprocessing
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_STORAGE = os.path.join(ROOT_DIR,"data", "results")
ALGORITHMIC_DATA_STORAGE = os.path.join(ROOT_DIR,"data", "algorithms")
# git test

id_changed = True
use_gradient_descent = True




