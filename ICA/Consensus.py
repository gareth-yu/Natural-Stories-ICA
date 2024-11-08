import os
import pandas as pd
import numpy as np
from utils_nmf import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
