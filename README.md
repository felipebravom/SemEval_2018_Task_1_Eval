# SemEval-2018 Task 1 Affect in Tweets Evaluation Script
Evaluation Script for [SemEval-2018](http://alt.qcri.org/semeval2018/) Task 1: [Affect in Tweets Task](http://www.saifmohammad.com/WebPages/affectintweets.htm).  


## 1. Evaluation Script
The evaluation script [evaluate.py](evaluate.py) calculates the performance metrics between the gold standard scores and the given predictions. 


### 1.1. Prerequisites
* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org)
### 1.2. Usage


 ```bash
python evaluate.py <task_type> <file-predictions> <file-gold> 
```
task_type: 1 for regression (EI-reg, V-reg), 2 for ordinal classification (EC-oc, V-oc), and 3 for multi-label classification (E-C).




### 1.3. Format
Each input file must have the format specified in the competition's [website](http://www.saifmohammad.com/WebPages/affectintweets.htm).  

If you want to use the script purely a format checker, evaluate your predictions against themselves:

 ```bash
python evaluate.py 1 EI-reg_en_fear_pred.txt EI-reg_en_fear_pred.txt
```


