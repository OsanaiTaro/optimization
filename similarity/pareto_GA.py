class genom:

    genom_list = None
#    evaluation_1 = None
#    evaluation_2 = None
#    rank = None

    def __init__(self, genom_list, evaluation_1):
        self.genom_list = genom_list
        self.evaluation_1 = evaluation_1
#        self.evaluation_2 = evaluation_2
#        self.rank = rank

    def getGenom(self):
        return self.genom_list

    def getEvaluation_1(self):
        return self.evaluation_1
        
#    def getEvaluation_2(self):
#        return self.evaluation_2
        
#    def getRank(self):
#        return self.rank


    def setGenom(self, genom_list):
        self.genom_list = genom_list

    def setEvaluation_1(self, evaluation_1):
        self.evaluation_1 = evaluation_1
        
#    def setEvaluation_2(self, evaluation_2):
#        self.evaluation_2 = evaluation_2
        
#    def setRank(self, rank):
#        self.rank = rank
'''
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage
from IPython.display import Image, display_png
import pandas as pd
import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

#plt.gray()
'''
