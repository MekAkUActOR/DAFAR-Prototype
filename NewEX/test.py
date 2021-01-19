#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:39:06 2020

@author: hongxing
"""

import sys
import argparse

parser = argparse.ArgumentParser()
parser.description='configuration'
parser.add_argument("-i", "--input", help="path of input picture", required=True)
parser.add_argument("-t", "--threshold", help="anomaly score threshold", type=float, required=True)
args = parser.parse_args()

print(args)

