# SemEval-2018 Task 1 Affect in Tweets Evaluation Script
Evaluation Script for [SemEval-2018](http://alt.qcri.org/semeval2018/) Task 1: [Affect in Tweets Task](http://www.saifmohammad.com/WebPages/affectintweets.htm).  


## 1. Evaluation Script
The evaluation script [evaluate.py](evaluate.py) calculates the following four measures between the gold standard scores and the given predictions. 
 1. Pearson_0_1: corresponds to the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) caclualted between all the predicions and the gold instances.
 2. Spearman_0_1: corresponds to the [Spearman's rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) caclualted between all the predicions and the gold instances.
 3. Pearson_05_1: corresponds to the Pearson correlation between predictions and gold instances where the gold score is greater than or equal to 0.5. 
 4. Separman_05_1: corresponds to the Spearman correlation between predictions and gold instances where the gold score is greater than or equal to 0.5. 
 
The script can receive multiple pairs of prediction and gold standard files. In case of receiving more than one pair, it will compute the average Pearson and Spearman correlation. Note that the average Pearson_0_1 obtained for the four emotions (anger, fear, joy and sadness) is the bottom-line competition metric and that remainder metrics are only given as a reference.


### 1.1. Prerequisites
* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org)
### 1.2. Usage


 ```bash
python evaluate.py <task_type> <file-predictions> <file-gold> 
```
task_type: 1 for regression, 2 for ordinal classification, and 3 for multi-label classification




### 1.3. Format
Each input file must have the format specified in the competition's [website](http://www.saifmohammad.com/WebPages/affectintweets.htm).  

If you want to use the script purely a format checker, evaluate your predictions against themselves:

 ```bash
python evaluate.py 1 EI-reg_en_fear_pred.txt EI-reg_en_fear_pred.txt
```


