#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# user_evaluation.py
# Author: felipebravom
# Descrition: Command-line version of the evaluation script for SemEval-2018 Task 1: Affect in Tweets
# usage: python evaluate.py <task_type> <file-predictions> <file-gold> 
# task_type: 1 for regression, 2 for ordinal classification, and 3 for multi-label emotion classification
# requires: numpy, scipy, sklearn


import sys
import os.path
from utils import evaluate_ei
from utils import evaluate_oc
from utils import evaluate_multilabel    



def main(argv):
    """main method """   
    
    if len(argv)!=3:
        raise ValueError('Invalid number of parameters.')


    task_type=int(argv[0])
    pred=argv[1]
    gold=argv[2] 

    if(task_type==1):
        result=evaluate_ei(pred,gold)
        print "Pearson correlation between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[0]) 
        print "Pearson correlation for gold scores in range 0.5-1 between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[1]) 
        
    elif(task_type==2):
        result=evaluate_oc(pred,gold)
        print "Pearson correlation between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[0]) 
        print "Pearson correlation for some emotions between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[1]) 
        print "Weighted quadratic Kappa between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[2]) 
        print "Weighted quadratic Kappa for some emotions between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[3]) 
        
        
    else:
        result=evaluate_multilabel(pred,gold)
        print "Multi-label accuracy (Jaccard index) between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[0]) 
        print "Micro-averaged F1 score between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[1]) 
        print "Macro-averaged F1 score between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[2]) 



if __name__ == "__main__":
    main(sys.argv[1:])