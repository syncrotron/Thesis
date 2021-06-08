#!/usr/bin/env python
# Developed by Samuel Horovatin at the University of Saskatchewan
# May, 2021

import os
import plantcv

class options:
    def __init__(self):
        # Input image path/filename
        self.image = "data/test_wheat_2021"
        # Debug mode = None, "plot", or "print"
        self.debug = "plot"
        # Store output images (True/False)
        self.writeimg = False
        # Results path/filename
        self.result = "results.txt"
        # Image output directory path
        self.outdir = "."

# Initialize options
args = options()