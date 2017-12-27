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

# utils.py
# Author: felipebravom
# Descrition: evaluation functions for SemEval-2018 Task 1: Affect in Tweets
# requires: numpy, scipy, sklearn



import sys
import os.path
import scipy.stats
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, jaccard_similarity_score


def cohen_kappa_score(y1, y2, labels=None, weights=None):
    """Cohen's kappa: a statistic that measures inter-annotator agreement.

    This function computes Cohen's kappa [1]_, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement on the label
    assigned to any sample (the observed agreement ratio), and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly.
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels [2]_.

    Read more in the :ref:`User Guide <cohen_kappa>`.

    Parameters
    ----------
    y1 : array, shape = [n_samples]
        Labels assigned by the first annotator.

    y2 : array, shape = [n_samples]
        Labels assigned by the second annotator. The kappa statistic is
        symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.

    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to select a
        subset of labels. If None, all labels that appear at least once in
        ``y1`` or ``y2`` are used.

    weights : str, optional
        List of weighting type to calculate the score. None means no weighted;
        "linear" means linear weighted; "quadratic" means quadratic weighted.

    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.

    References
    ----------
    .. [1] J. Cohen (1960). "A coefficient of agreement for nominal scales".
           Educational and Psychological Measurement 20(1):37-46.
           doi:10.1177/001316446002000104.
    .. [2] `R. Artstein and M. Poesio (2008). "Inter-coder agreement for
           computational linguistics". Computational Linguistics 34(4):555-596.
           <http://www.mitpressjournals.org/doi/abs/10.1162/coli.07-034-R2#.V0J1MJMrIWo>`_
    .. [3] `Wikipedia entry for the Cohen's kappa.
            <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_
    """
    confusion = metrics.confusion_matrix(y1, y2, labels=labels)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1)*1.0 / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


def evaluate_ei(pred,gold):  
    """Calculates performance metrics for regression.
    
    :param pred: the file path of the predictions
    :param gold: the filte path withe gold data
    :return: a list with performace metrics.
    
    """
    
    f=open(pred, "rb")
    pred_lines=f.readlines()
    f.close()
    
    f=open(gold, "rb")
    gold_lines=f.readlines()
    f.close()    

    if(len(pred_lines)==len(gold_lines)):       
        # align tweets ids with gold scores and predictions
        data_dic={}


        header=True        
        for line in gold_lines:
            #line=line.decode('utf-8')
            
            if header:
                header=False
                continue

            parts=line.split('\t')
            if len(parts)==4:
                # tweet ids containing the word mystery are discarded
                if(not 'mystery' in parts[0]):
                    data_dic[parts[0]]=[float(line.split('\t')[3])]
            else:
                sys.exit('Format problem in '+os.path.basename(gold)+'. Please report this problem to the task organizers.')
                
 

        header=True        
        for line in pred_lines:
            
            if header:
                header=False
                continue

            parts=line.split('\t')      
            if len(parts)==4:
                # tweet ids containing the word mystery are discarded
                if(not 'mystery' in parts[0]):
                    if parts[0] in data_dic:
                        try:
                            data_dic[parts[0]].append(float(line.split('\t')[3]))
                        except ValueError:
                            # Invalid predictions are replaced by a default value
                            data_dic[parts[0]].append(0.5)
                    else:
                        sys.exit('Invalid tweet id ('+parts[0]+') in '+os.path.basename(pred)+'.')
            else:
                sys.exit('Format problem in '+os.path.basename(pred)+'.') 
            
        
        # lists storing gold and prediction scores
        gold_scores=[]  
        pred_scores=[]
         
        
        # lists storing gold and prediction scores where gold score >= 0.5
        gold_scores_range_05_1=[]
        pred_scores_range_05_1=[]
            
        for id in data_dic:
            if(len(data_dic[id])==2):
                gold_scores.append(data_dic[id][0])
                pred_scores.append(data_dic[id][1])
                
                if(data_dic[id][0]>=0.5):
                    gold_scores_range_05_1.append(data_dic[id][0])
                    pred_scores_range_05_1.append(data_dic[id][1])
                
            else:
                sys.exit('Repeated id ('+id+') in '+os.path.basename(pred)+' .')

                
                    
      
        # return zero correlation if predictions are constant
        if np.std(pred_scores)==0 or np.std(gold_scores)==0:
            return (0,0)
        

        pears_corr=scipy.stats.pearsonr(pred_scores,gold_scores)[0]                                                 
                                    
        
        pears_corr_range_05_1=scipy.stats.pearsonr(pred_scores_range_05_1,gold_scores_range_05_1)[0]                         
        
      
        return (pears_corr,pears_corr_range_05_1)
       
                                    
                          
        
    else:
        sys.exit('Predictions ('+os.path.basename(pred)+') and gold data ('+os.path.basename(gold)+') have different number of lines.')          
  
        
        
def evaluate_oc(pred,gold):  
    """Calculates performance metrics for ordinal classification.
    
    :param pred: the file path of the predictions
    :param gold: the filte path withe gold data
    :return: a list with performace metrics.
    
    """
  
   
    f=open(pred, "rb")
    pred_lines=f.readlines()
    f.close()
    
    f=open(gold, "rb")
    gold_lines=f.readlines()
    f.close()    


    if(len(pred_lines)==len(gold_lines)):       
        # align tweets ids with gold scores and predictions
        data_dic={}

    
        header=True        
        for line in gold_lines:
            
            if header:
                header=False
                continue
            
            parts=line.split('\t')
            
            label=int(parts[3].split(":")[0])
            
            if len(parts)==4:   
                data_dic[parts[0]]=[label]
                
            else:
                sys.exit('Format problem in '+os.path.basename(gold)+'. Please report this problem to the task organizers.')
        
        
        header=True         
        for line in pred_lines:
            if header:
                header=False
                continue            
            parts=line.split('\t')   
            label=int(parts[3].split(":")[0])
            if len(parts)==4:  
                if parts[0] in data_dic:
                    try:
                        data_dic[parts[0]].append(label)
                    except ValueError:
                        # Invalid predictions are replaced by a default value
                        data_dic[parts[0]].append(int(0))
                else:
                    sys.exit('Invalid tweet id ('+parts[0]+') in '+os.path.basename(pred)+'.')
            else:
                sys.exit('Format problem in '+os.path.basename(pred)+'.') 
            
        
        # lists storing gold and prediction scores
        gold_scores=[]  
        pred_scores=[]
         
        
        # lists storing gold and prediction scores where gold score >= 0.5
        gold_scores_some=[]
        pred_scores_some=[]
            
        for id in data_dic:
            if(len(data_dic[id])==2):
                gold_scores.append(data_dic[id][0])
                pred_scores.append(data_dic[id][1])
                
                if(data_dic[id][0]!=0):
                    gold_scores_some.append(data_dic[id][0])
                    pred_scores_some.append(data_dic[id][1])
                
            else:
                sys.exit('Repeated id ('+id+') in '+os.path.basename(pred)+' .')

                
                
        # return null scores if predictions are constant
        if np.std(pred_scores)==0 or np.std(gold_scores)==0:
            return (0,0,0,0)
        

        pears_corr=scipy.stats.pearsonr(pred_scores,gold_scores)[0]              
        pears_corr_some=scipy.stats.pearsonr(pred_scores_some,gold_scores_some)[0]  
        

        # fix labels to values observed in gold data        
        gold_labels=list(sorted(set(gold_scores)))
      

       
        kappa=cohen_kappa_score(pred_scores,gold_scores,labels=gold_labels, weights='quadratic')        
        kappa_some=cohen_kappa_score(pred_scores_some,gold_scores_some, labels=gold_labels, weights='quadratic')
                                                    
           
        return (pears_corr,pears_corr_some,kappa,kappa_some)
       
                                    
                          
        
    else:
        sys.exit('Predictions ('+os.path.basename(pred)+') and gold data ('+os.path.basename(gold)+') have different number of lines.')          
  
        
        
        
        
        
        
def evaluate_multilabel(pred,gold):  
    """Calculates performance metrics for multi-label classification.
    
    :param pred: the file path of the predictions
    :param gold: the filte path withe gold data
    :return: a list with performace metrics.    
    """     
    
    f=open(pred, "rb")
    pred_lines=f.readlines()
    f.close()
    
    f=open(gold, "rb")
    gold_lines=f.readlines()
    f.close()    

    if(len(pred_lines)==len(gold_lines)):       
        # align tweets ids with gold scores and predictions
        data_dic={}

    
        header=True        
        for line in gold_lines:
           
            if header:
                header=False
                continue
            
            parts=line.split('\t')
            

            
            if len(parts)==13:  
                labels=[]
                for m_label in parts[2:13]:
                    labels.append(int(m_label))
 
                data_dic[parts[0]]=[tuple(labels)]
                
            else:
                sys.exit('Format problem in '+os.path.basename(gold)+'. Please report this problem to the task organizers.')
        
        header=True         
        for line in pred_lines:
            if header:
                header=False
                continue            
            parts=line.split('\t')   
            if len(parts)==13:  
                if parts[0] in data_dic:
                    try:
                        labels=[]
                        for m_label in parts[2:13]:
                            labels.append(int(m_label))
                        data_dic[parts[0]].append(tuple(labels))
                    except ValueError:
                        # Invalid predictions are replaced by a default value
                        data_dic[parts[0]].append((0,0,0,0,0,0,0,0,0,0,0))
                else:
                    sys.exit('Invalid tweet id in '+os.path.basename(pred)+'.')
            else:
                sys.exit('Format problem in '+os.path.basename(pred)+'.')   
            
            


       # lists storing gold and prediction scores
        gold_scores=[]  
        pred_scores=[]
         
            
        for id in data_dic:
            if(len(data_dic[id])==2):
                gold_scores.append(data_dic[id][0])
                pred_scores.append(data_dic[id][1])
                
  
                
            else:
                sys.exit('Repeated id ('+id+') in '+os.path.basename(pred)+' .')



        y_true = np.array(gold_scores)
        y_pred = np.array(pred_scores)       
    
        acc=jaccard_similarity_score(y_true,y_pred)       
    
        f1_micro=f1_score(y_true, y_pred, average='micro')  
    
        f1_macro=f1_score(y_true, y_pred, average='macro')  

            
        return (acc,f1_micro,f1_macro)
    
    else:
        sys.exit('Predictions ('+os.path.basename(pred)+') and gold data ('+os.path.basename(gold)+') have different number of lines.')          
  