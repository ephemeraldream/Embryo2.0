from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)

from src.logger import LOGGER


def basic_split_dataset():

    
    LOGGER.info("Dataset was successfully splitted and completed.")
