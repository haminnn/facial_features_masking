from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
import os
import facial_features_masking
from pathlib import Path


facial_features_masking.eyebrows_masking_call()
facial_features_masking.eyes_masking_call()
facial_features_masking.nose_masking_call()
facial_features_masking.mouth_masking_call()

