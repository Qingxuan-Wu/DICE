"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Basic logger. It Computes and stores the average and current value
"""

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class EvalMetricsLogger(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        # define a upper-bound performance (worst case) 
        # numbers are in unit millimeter
        self.PAmPJPE = 100.0/1000.0
        self.mPJPE = 100.0/1000.0
        self.mPVE = 100.0/1000.0
        self.mPVE_trans = 100.0/1000.0
        self.ColDist = 100.0/1000.0
        self.Touchness = 0
        self.NonColRatio = 0
        self.FScore = 0

        self.epoch = 0

    def update(self, mPVE=None, mPJPE=None, PAmPJPE=None, mPVE_trans=None, ColDist=None, Touchness=None, NonColRatio=None, FScore=None, epoch=None):
        self.PAmPJPE = PAmPJPE
        self.mPJPE = mPJPE
        self.mPVE = mPVE
        self.mPVE_trans = mPVE_trans
        self.ColDist = ColDist
        self.Touchness = Touchness
        self.NonColRatio = NonColRatio
        self.FScore = FScore
