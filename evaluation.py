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

# evaluation.py
# Author: felipebravom
# Descrition: Codalab's evaluation script for SemEval-2018 Task 1: Affect in Tweets
# requires: numpy, scipy, sklearn

import sys
import os.path
import numpy
from utils import evaluate_ei
from utils import evaluate_oc
from utils import evaluate_multilabel       
        


def find(name, path):    
    """ finds a filename in a folder using a recursive search """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        else:
            if len(dirs)>0:
                for fold in dirs:
                    find(name,os.path.join(root,fold))



def check_files(input_dir):
    """checks submitted files for all tasks"""
    
    submissions={}
    
    
    
    # EI-REG-EN
    submissions['ei-reg-en-anger']=find('EI-reg_en_anger_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-en-fear']= find('EI-reg_en_fear_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-en-joy']=find('EI-reg_en_joy_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-en-sadness']= find('EI-reg_en_sadness_pred.txt',os.path.join(input_dir, 'res'))
   
    # EI-REG-AR    
    submissions['ei-reg-ar-anger']=find('EI-reg_ar_anger_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-ar-fear']= find('EI-reg_ar_fear_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-ar-joy']=find('EI-reg_ar_joy_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-ar-sadness']= find('EI-reg_ar_sadness_pred.txt',os.path.join(input_dir, 'res'))
    
    # EI-REG-ES    
    submissions['ei-reg-es-anger']=find('EI-reg_es_anger_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-es-fear']= find('EI-reg_es_fear_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-es-joy']=find('EI-reg_es_joy_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-reg-es-sadness']= find('EI-reg_es_sadness_pred.txt',os.path.join(input_dir, 'res'))    
    

    # EI-OC-EN
    submissions['ei-oc-en-anger']=find('EI-oc_en_anger_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-en-fear']= find('EI-oc_en_fear_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-en-joy']=find('EI-oc_en_joy_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-en-sadness']= find('EI-oc_en_sadness_pred.txt',os.path.join(input_dir, 'res'))


    # EI-OC-AR
    submissions['ei-oc-ar-anger']=find('EI-oc_ar_anger_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-ar-fear']= find('EI-oc_ar_fear_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-ar-joy']=find('EI-oc_ar_joy_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-ar-sadness']= find('EI-oc_ar_sadness_pred.txt',os.path.join(input_dir, 'res'))

    # EI-OC-ES
    submissions['ei-oc-es-anger']=find('EI-oc_es_anger_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-es-fear']= find('EI-oc_es_fear_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-es-joy']=find('EI-oc_es_joy_pred.txt',os.path.join(input_dir, 'res'))
    submissions['ei-oc-es-sadness']= find('EI-oc_es_sadness_pred.txt',os.path.join(input_dir, 'res'))
    
    
    # V-REG    
    submissions['v-reg-en']=find('V-reg_en_pred.txt',os.path.join(input_dir, 'res'))
    submissions['v-reg-ar']=find('V-reg_ar_pred.txt',os.path.join(input_dir, 'res'))
    submissions['v-reg-es']=find('V-reg_es_pred.txt',os.path.join(input_dir, 'res'))    
    
    # V-OC  
    submissions['v-oc-en']=find('V-oc_en_pred.txt',os.path.join(input_dir, 'res'))
    submissions['v-oc-ar']=find('V-oc_ar_pred.txt',os.path.join(input_dir, 'res'))
    submissions['v-oc-es']=find('V-oc_es_pred.txt',os.path.join(input_dir, 'res'))  
    
    
    # E-C   
    submissions['e-c-en']=find('E-C_en_pred.txt',os.path.join(input_dir, 'res'))
    submissions['e-c-ar']=find('E-C_ar_pred.txt',os.path.join(input_dir, 'res'))
    submissions['e-c-es']=find('E-C_es_pred.txt',os.path.join(input_dir, 'res'))
    

    
    return submissions


def init_ref_files(input_dir):
    """ declares paths for gold files"""
    ref_files={}
    ref_files['ei_reg_en_anger'] = os.path.join(input_dir, 'ref', '2018-EI-reg-En-anger-gold.txt')
    ref_files['ei_reg_en_fear']  = os.path.join(input_dir, 'ref', '2018-EI-reg-En-fear-gold.txt')
    ref_files['ei_reg_en_joy'] = os.path.join(input_dir, 'ref', '2018-EI-reg-En-joy-gold.txt')
    ref_files['ei_reg_en_sadness'] = os.path.join(input_dir, 'ref', '2018-EI-reg-En-sadness-gold.txt')
    
    ref_files['ei_reg_ar_anger'] = os.path.join(input_dir, 'ref', '2018-EI-reg-Ar-anger-gold.txt')
    ref_files['ei_reg_ar_fear'] = os.path.join(input_dir, 'ref', '2018-EI-reg-Ar-fear-gold.txt')
    ref_files['ei_reg_ar_joy'] = os.path.join(input_dir, 'ref', '2018-EI-reg-Ar-joy-gold.txt')
    ref_files['ei_reg_ar_sadness'] = os.path.join(input_dir, 'ref', '2018-EI-reg-Ar-sadness-gold.txt')
    

    ref_files['ei_reg_es_anger'] = os.path.join(input_dir, 'ref', '2018-EI-reg-Es-anger-gold.txt')
    ref_files['ei_reg_es_fear'] = os.path.join(input_dir, 'ref', '2018-EI-reg-Es-fear-gold.txt')
    ref_files['ei_reg_es_joy'] = os.path.join(input_dir, 'ref', '2018-EI-reg-Es-joy-gold.txt')
    ref_files['ei_reg_es_sadness'] = os.path.join(input_dir, 'ref', '2018-EI-reg-Es-sadness-gold.txt')


    ref_files['ei_oc_en_anger'] = os.path.join(input_dir, 'ref', '2018-EI-oc-En-anger-gold.txt')
    ref_files['ei_oc_en_fear']  = os.path.join(input_dir, 'ref', '2018-EI-oc-En-fear-gold.txt')
    ref_files['ei_oc_en_joy'] = os.path.join(input_dir, 'ref', '2018-EI-oc-En-joy-gold.txt')
    ref_files['ei_oc_en_sadness'] = os.path.join(input_dir, 'ref', '2018-EI-oc-En-sadness-gold.txt')
    
    ref_files['ei_oc_ar_anger'] = os.path.join(input_dir, 'ref', '2018-EI-oc-Ar-anger-gold.txt')
    ref_files['ei_oc_ar_fear'] = os.path.join(input_dir, 'ref', '2018-EI-oc-Ar-fear-gold.txt')
    ref_files['ei_oc_ar_joy'] = os.path.join(input_dir, 'ref', '2018-EI-oc-Ar-joy-gold.txt')
    ref_files['ei_oc_ar_sadness'] = os.path.join(input_dir, 'ref', '2018-EI-oc-Ar-sadness-gold.txt')
    

    ref_files['ei_oc_es_anger'] = os.path.join(input_dir, 'ref', '2018-EI-oc-Es-anger-gold.txt')
    ref_files['ei_oc_es_fear'] = os.path.join(input_dir, 'ref', '2018-EI-oc-Es-fear-gold.txt')
    ref_files['ei_oc_es_joy'] = os.path.join(input_dir, 'ref', '2018-EI-oc-Es-joy-gold.txt')
    ref_files['ei_oc_es_sadness'] = os.path.join(input_dir, 'ref', '2018-EI-oc-Es-sadness-gold.txt')
    
    
    ref_files['valence_reg_en'] = os.path.join(input_dir, 'ref', '2018-Valence-reg-En-gold.txt')
    ref_files['valence_reg_ar'] = os.path.join(input_dir, 'ref', '2018-Valence-reg-Ar-gold.txt')
    ref_files['valence_reg_es'] = os.path.join(input_dir, 'ref', '2018-Valence-reg-Es-gold.txt')

    ref_files['valence_oc_en'] = os.path.join(input_dir, 'ref', '2018-Valence-oc-En-gold.txt')
    ref_files['valence_oc_ar'] = os.path.join(input_dir, 'ref', '2018-Valence-oc-Ar-gold.txt')
    ref_files['valence_oc_es'] = os.path.join(input_dir, 'ref', '2018-Valence-oc-Es-gold.txt')


    ref_files['e-c_en'] = os.path.join(input_dir, 'ref', '2018-E-c-En-gold.txt')
    ref_files['e-c_ar'] = os.path.join(input_dir, 'ref', '2018-E-c-Ar-gold.txt')
    ref_files['e-c_es'] = os.path.join(input_dir, 'ref', '2018-E-c-Es-gold.txt')

    



    return ref_files

def subtasks(submissions):
    """ return the substasks with valid submissions """
    results={}
    
    
    results["ei-reg-en"] = submissions['ei-reg-en-anger'] is not None and submissions['ei-reg-en-fear'] is not None and submissions['ei-reg-en-joy'] is not None and submissions['ei-reg-en-sadness'] is not None 
    results["ei-reg-ar"] = submissions['ei-reg-ar-anger'] is not None and submissions['ei-reg-ar-fear'] is not None and submissions['ei-reg-ar-joy'] is not None and submissions['ei-reg-ar-sadness'] is not None 
    results["ei-reg-es"] = submissions['ei-reg-es-anger'] is not None and submissions['ei-reg-es-fear'] is not None and submissions['ei-reg-es-joy'] is not None and submissions['ei-reg-es-sadness'] is not None 
   

    results["ei-oc-en"] = submissions['ei-oc-en-anger'] is not None and submissions['ei-oc-en-fear'] is not None and submissions['ei-oc-en-joy'] is not None and submissions['ei-oc-en-sadness'] is not None 
    results["ei-oc-ar"] = submissions['ei-oc-ar-anger'] is not None and submissions['ei-oc-ar-fear'] is not None and submissions['ei-oc-ar-joy'] is not None and submissions['ei-oc-ar-sadness'] is not None 
    results["ei-oc-es"] = submissions['ei-oc-es-anger'] is not None and submissions['ei-oc-es-fear'] is not None and submissions['ei-oc-es-joy'] is not None and submissions['ei-oc-es-sadness'] is not None 
 

    results["v-reg-en"] = submissions['v-reg-en'] is not None 
    results["v-reg-ar"] = submissions['v-reg-ar'] is not None
    results["v-reg-es"] = submissions['v-reg-es'] is not None

    results["v-oc-en"] = submissions['v-oc-en'] is not None 
    results["v-oc-ar"] = submissions['v-oc-ar'] is not None
    results["v-oc-es"] = submissions['v-oc-es'] is not None

    results["e-c-en"] = submissions['e-c-en'] is not None 
    results["e-c-ar"] = submissions['e-c-ar'] is not None 
    results["e-c-es"] = submissions['e-c-es'] is not None            
    

          
    return results



def init_metrics():
    """ initiliazes a dictionary with all the metrics """
    metrics={}
    
    metrics['r_ei_macro_avg_en']=-999.999    
    metrics['r_ei_anger_en']=-999.999
    metrics['r_ei_fear_en']=-999.999
    metrics['r_ei_joy_en']=-999.999
    metrics['r_ei_sadness_en']=-999.999  
    
    metrics['r_05_1_ei_macro_avg_en']=-999.999    
    metrics['r_05_1_ei_anger_en']=-999.999
    metrics['r_05_1_ei_fear_en']=-999.999
    metrics['r_05_1_ei_joy_en']=-999.999
    metrics['r_05_1_ei_sadness_en']=-999.999      
    

    metrics['r_ei_macro_avg_ar']=-999.999    
    metrics['r_ei_anger_ar']=-999.999
    metrics['r_ei_fear_ar']=-999.999
    metrics['r_ei_joy_ar']=-999.999
    metrics['r_ei_sadness_ar']=-999.999
    
    
    metrics['r_05_1_ei_macro_avg_ar']=-999.999    
    metrics['r_05_1_ei_anger_ar']=-999.999
    metrics['r_05_1_ei_fear_ar']=-999.999
    metrics['r_05_1_ei_joy_ar']=-999.999
    metrics['r_05_1_ei_sadness_ar']=-999.999    

    
    metrics['r_ei_macro_avg_es']=-999.999    
    metrics['r_ei_anger_es']=-999.999
    metrics['r_ei_fear_es']=-999.999
    metrics['r_ei_joy_es']=-999.999
    metrics['r_ei_sadness_es']=-999.999


    metrics['r_05_1_ei_macro_avg_es']=-999.999    
    metrics['r_05_1_ei_anger_es']=-999.999
    metrics['r_05_1_ei_fear_es']=-999.999
    metrics['r_05_1_ei_joy_es']=-999.999
    metrics['r_05_1_ei_sadness_es']=-999.999
    
    
    metrics['r_oc_macro_avg_en']=-999.999    
    metrics['r_oc_anger_en']=-999.999
    metrics['r_oc_fear_en']=-999.999
    metrics['r_oc_joy_en']=-999.999
    metrics['r_oc_sadness_en']=-999.999  
    
    
    metrics['r_some_oc_macro_avg_en']=-999.999    
    metrics['r_some_oc_anger_en']=-999.999
    metrics['r_some_oc_fear_en']=-999.999
    metrics['r_some_oc_joy_en']=-999.999
    metrics['r_some_oc_sadness_en']=-999.999  
    
    
    metrics['kappa_oc_macro_avg_en']=-999.999    
    metrics['kappa_oc_anger_en']=-999.999
    metrics['kappa_oc_fear_en']=-999.999
    metrics['kappa_oc_joy_en']=-999.999
    metrics['kappa_oc_sadness_en']=-999.999  
    
    
    metrics['kappa_some_oc_macro_avg_en']=-999.999    
    metrics['kappa_some_oc_anger_en']=-999.999
    metrics['kappa_some_oc_fear_en']=-999.999
    metrics['kappa_some_oc_joy_en']=-999.999
    metrics['kappa_some_oc_sadness_en']=-999.999  
    
    
    

    metrics['r_oc_macro_avg_ar']=-999.999    
    metrics['r_oc_anger_ar']=-999.999
    metrics['r_oc_fear_ar']=-999.999
    metrics['r_oc_joy_ar']=-999.999
    metrics['r_oc_sadness_ar']=-999.999


    metrics['r_some_oc_macro_avg_ar']=-999.999    
    metrics['r_some_oc_anger_ar']=-999.999
    metrics['r_some_oc_fear_ar']=-999.999
    metrics['r_some_oc_joy_ar']=-999.999
    metrics['r_some_oc_sadness_ar']=-999.999 


    metrics['kappa_oc_macro_avg_ar']=-999.999    
    metrics['kappa_oc_anger_ar']=-999.999
    metrics['kappa_oc_fear_ar']=-999.999
    metrics['kappa_oc_joy_ar']=-999.999
    metrics['kappa_oc_sadness_ar']=-999.999  
    
   
    metrics['kappa_some_oc_macro_avg_ar']=-999.999    
    metrics['kappa_some_oc_anger_ar']=-999.999
    metrics['kappa_some_oc_fear_ar']=-999.999
    metrics['kappa_some_oc_joy_ar']=-999.999
    metrics['kappa_some_oc_sadness_ar']=-999.999  

  
    metrics['r_oc_macro_avg_es']=-999.999    
    metrics['r_oc_anger_es']=-999.999
    metrics['r_oc_fear_es']=-999.999
    metrics['r_oc_joy_es']=-999.999
    metrics['r_oc_sadness_es']=-999.999    
    
    
    metrics['r_some_oc_macro_avg_es']=-999.999    
    metrics['r_some_oc_anger_es']=-999.999
    metrics['r_some_oc_fear_es']=-999.999
    metrics['r_some_oc_joy_es']=-999.999
    metrics['r_some_oc_sadness_es']=-999.999     
    
    metrics['kappa_oc_macro_avg_es']=-999.999    
    metrics['kappa_oc_anger_es']=-999.999
    metrics['kappa_oc_fear_es']=-999.999
    metrics['kappa_oc_joy_es']=-999.999
    metrics['kappa_oc_sadness_es']=-999.999  
    
    
    metrics['kappa_some_oc_macro_avg_es']=-999.999    
    metrics['kappa_some_oc_anger_es']=-999.999
    metrics['kappa_some_oc_fear_es']=-999.999
    metrics['kappa_some_oc_joy_es']=-999.999
    metrics['kappa_some_oc_sadness_es']=-999.999  
    
    metrics['r_reg_valence_en']=-999.999  
    metrics['r_05_1_reg_valence_en']=-999.999  
   
    
    metrics['r_reg_valence_ar']=-999.999
    metrics['r_05_1_reg_valence_ar']=-999.999  
    
    metrics['r_reg_valence_es']=-999.999
    metrics['r_05_1_reg_valence_es']=-999.999  
    


    metrics['r_oc_valence_en']=-999.999 
    metrics['r_some_oc_valence_en']=-999.999                
    metrics['kappa_oc_valence_en']=-999.999  
    metrics['kappa_some_oc_valence_en']=-999.999              
    
    metrics['r_oc_valence_ar']=-999.999 
    metrics['r_some_oc_valence_ar']=-999.999                
    metrics['kappa_oc_valence_ar']=-999.999  
    metrics['kappa_some_oc_valence_ar']=-999.999 
    
    
    metrics['r_oc_valence_es']=-999.999 
    metrics['r_some_oc_valence_es']=-999.999                
    metrics['kappa_oc_valence_es']=-999.999  
    metrics['kappa_some_oc_valence_es']=-999.999 
    
 
    
    metrics['acc_e-c_en']=-999.999   
    metrics['f1_micro_e-c_en']=-999.999   
    metrics['f1_macro_e-c_en']=-999.999  
    
    metrics['acc_e-c_ar']=-999.999   
    metrics['f1_micro_e-c_ar']=-999.999   
    metrics['f1_macro_e-c_ar']=-999.999   
    
    
    metrics['acc_e-c_es']=-999.999   
    metrics['f1_micro_e-c_es']=-999.999   
    metrics['f1_macro_e-c_es']=-999.999      
    

    
    return metrics




def write_metrics(metrics,output_file):
    """ writes output for Codalab """
    for keys in metrics.iterkeys():
        output_file.write(keys+":{0}\n".format(metrics[keys]))     

        
  
def ei_reg_scores(lang,submissions,metrics,ref_files):
    """calculates metrics for ei-reg tasks """
    
    anger_scores=evaluate_ei(submissions['ei-reg-'+lang+'-anger'], ref_files['ei_reg_'+lang+'_anger'])
    fear_scores=evaluate_ei(submissions['ei-reg-'+lang+'-fear'], ref_files['ei_reg_'+lang+'_fear'])    
    joy_scores=evaluate_ei(submissions['ei-reg-'+lang+'-joy'], ref_files['ei_reg_'+lang+'_joy'] )
    sadness_scores=evaluate_ei(submissions['ei-reg-'+lang+'-sadness'], ref_files['ei_reg_'+lang+'_sadness'])      

       
    metrics['r_ei_anger_'+lang]=anger_scores[0]
    metrics['r_ei_fear_'+lang]=fear_scores[0]
    metrics['r_ei_joy_'+lang]=joy_scores[0]
    metrics['r_ei_sadness_'+lang]=sadness_scores[0]
    metrics['r_ei_macro_avg_'+lang]=numpy.mean([anger_scores[0],fear_scores[0],joy_scores[0],sadness_scores[0]])
    

    metrics['r_05_1_ei_anger_'+lang]=anger_scores[1]
    metrics['r_05_1_ei_fear_'+lang]=fear_scores[1]
    metrics['r_05_1_ei_joy_'+lang]=joy_scores[1]
    metrics['r_05_1_ei_sadness_'+lang]=sadness_scores[1]
    metrics['r_05_1_ei_macro_avg_'+lang]=numpy.mean([anger_scores[1],fear_scores[1],joy_scores[1],sadness_scores[1]])    
          




def ei_oc_scores(lang,submissions,metrics,ref_files):
    """calculates metrics for ei-oc tasks """
    
    anger_scores=evaluate_oc(submissions['ei-oc-'+lang+'-anger'], ref_files['ei_oc_'+lang+'_anger'])
    fear_scores=evaluate_oc(submissions['ei-oc-'+lang+'-fear'], ref_files['ei_oc_'+lang+'_fear'])    
    joy_scores=evaluate_oc(submissions['ei-oc-'+lang+'-joy'], ref_files['ei_oc_'+lang+'_joy'] )
    sadness_scores=evaluate_oc(submissions['ei-oc-'+lang+'-sadness'], ref_files['ei_oc_'+lang+'_sadness'])      
     

    metrics['r_oc_anger_'+lang] = anger_scores[0]
    metrics['r_oc_fear_'+lang] = fear_scores[0]
    metrics['r_oc_joy_'+lang] = joy_scores[0]
    metrics['r_oc_sadness_'+lang] = sadness_scores[0]
    metrics['r_oc_macro_avg_'+lang] = numpy.mean([anger_scores[0],fear_scores[0],joy_scores[0],sadness_scores[0]])
    
       
    metrics['r_some_oc_anger_'+lang] = anger_scores[1]
    metrics['r_some_oc_fear_'+lang] = fear_scores[1]
    metrics['r_some_oc_joy_'+lang] = joy_scores[1]
    metrics['r_some_oc_sadness_'+lang] = sadness_scores[1]  
    metrics['r_some_oc_macro_avg_'+lang] = numpy.mean([anger_scores[1],fear_scores[1],joy_scores[1],sadness_scores[1]])       
    
    
    metrics['kappa_oc_anger_'+lang] = anger_scores[2]
    metrics['kappa_oc_fear_'+lang] = fear_scores[2]
    metrics['kappa_oc_joy_'+lang] = joy_scores[2]
    metrics['kappa_oc_sadness_'+lang] = sadness_scores[2]
    metrics['kappa_oc_macro_avg_'+lang] = numpy.mean([anger_scores[2],fear_scores[2],joy_scores[2],sadness_scores[2]])   
    
    
    metrics['kappa_some_oc_anger_'+lang] = anger_scores[3]
    metrics['kappa_some_oc_fear_'+lang]= fear_scores[3]
    metrics['kappa_some_oc_joy_'+lang]= joy_scores[3]
    metrics['kappa_some_oc_sadness_'+lang] = sadness_scores[3]
    metrics['kappa_some_oc_macro_avg_'+lang]= numpy.mean([anger_scores[3],fear_scores[3],joy_scores[3],sadness_scores[3]])   
     



def v_reg_scores(lang,submissions,metrics,ref_files):
    """calculates metrics for ei-reg valence tasks """
    
    valence_scores=evaluate_ei(submissions['v-reg-'+lang], ref_files['valence_reg_'+lang])
        
    metrics['r_reg_valence_'+lang]=valence_scores[0]    

    metrics['r_05_1_reg_valence_'+lang]=valence_scores[1]
    
    

def v_oc_scores(lang,submissions,metrics,ref_files):
    """calculates metrics for ei-reg valence tasks """
    
    valence_scores=evaluate_oc(submissions['v-oc-'+lang], ref_files['valence_oc_'+lang])
        
    metrics['r_oc_valence_'+lang] = valence_scores[0]
    metrics['r_some_oc_valence_'+lang] = valence_scores[1]
    metrics['kappa_oc_valence_'+lang] = valence_scores[2]
    metrics['kappa_some_oc_valence_'+lang] = valence_scores[3]




def multi_label_scores(lang,submissions,metrics,ref_files):
    """calculates metrics for ei-reg valence tasks """
    
    multi_label_scores=evaluate_multilabel(submissions['e-c-'+lang], ref_files['e-c_'+lang])
        
    metrics['acc_e-c_'+lang] = multi_label_scores[0]
    metrics['f1_micro_e-c_'+lang] = multi_label_scores[1]
    metrics['f1_macro_e-c_'+lang] = multi_label_scores[2]



def main(argv):
    #https://github.com/Tivix/competition-examples/blob/master/compute_pi/program/evaluate.py
    # as per the metadata file, input and output directories are the arguments

    [input_dir, output_dir] = argv
    


    ref_files=init_ref_files(input_dir)
    submissions=check_files(input_dir)
    
    tasks=subtasks(submissions)
    
    if( not (tasks['ei-reg-en'] or tasks['ei-reg-ar'] or tasks['ei-reg-es'] or tasks['ei-oc-en'] or tasks['ei-oc-ar'] or tasks['ei-oc-es'] or tasks['v-reg-en'] or tasks['v-reg-ar'] or tasks['v-reg-es'] or tasks['v-oc-en'] or tasks['v-oc-ar'] or tasks['v-oc-es'] or tasks['e-c-en'] or tasks['e-c-ar'] or tasks['e-c-es'])):
        sys.exit('Invalid combination of submitted files.')  
    
    
    
    
    
    
    
    metrics=init_metrics()    
    
    
    # unzipped submission data is always in the 'res' subdirectory
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    
#    input_dir='/Users/admin/Dropbox/SemEval2018/codalab/sample_sub/'
#    output_dir='/Users/admin/Dropbox/SemEval2018/codalab/sample_sub/'
    
    
    
    
    if(tasks['ei-reg-en']):            
        ei_reg_scores('en',submissions,metrics,ref_files)
    
    
    if(tasks['ei-reg-ar']):            
        ei_reg_scores('ar',submissions,metrics,ref_files)        
    
    
    if(tasks['ei-reg-es']):            
       ei_reg_scores('es',submissions,metrics,ref_files)
    
    
    # calculates ei-oc-en metrics
    
    if(tasks['ei-oc-en']):            
        ei_oc_scores('en',submissions,metrics,ref_files)


    if(tasks['ei-oc-ar']):            
        ei_oc_scores('ar',submissions,metrics,ref_files)


    if(tasks['ei-oc-es']):            
        ei_oc_scores('es',submissions,metrics,ref_files)
        
        
    if(tasks['v-reg-en']):            
        v_reg_scores('en',submissions,metrics,ref_files)       
        

    if(tasks['v-reg-ar']):            
        v_reg_scores('ar',submissions,metrics,ref_files) 
        

    if(tasks['v-reg-es']):            
        v_reg_scores('es',submissions,metrics,ref_files) 


    if(tasks['v-oc-en']):            
        v_oc_scores('en',submissions,metrics,ref_files)       
        

    if(tasks['v-oc-ar']):            
        v_oc_scores('ar',submissions,metrics,ref_files) 
        

    if(tasks['v-oc-es']):            
        v_oc_scores('es',submissions,metrics,ref_files) 


    if(tasks['e-c-en']):            
        multi_label_scores('en',submissions,metrics,ref_files) 

    if(tasks['e-c-ar']):            
        multi_label_scores('ar',submissions,metrics,ref_files) 

    if(tasks['e-c-es']):            
       multi_label_scores('es',submissions,metrics,ref_files) 


    
    # the scores for the leaderboard must be in a file named "scores.txt"
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    
    output_file=open(os.path.join(output_dir, 'scores.txt'),"w")

    write_metrics(metrics,output_file)
   
    
    
    
    output_file.close()
    
 
    
    
if __name__ == "__main__":
    main(sys.argv[1:])    
    
    