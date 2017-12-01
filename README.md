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

The parameter <task_type> can take three possible values:
* 1 for regression (EI-reg, V-reg)
* 2 for ordinal classification (EC-oc, V-oc)
* 3 for multi-label classification (E-C).



### 1.3. Format
Each input file must have the format specified in the competition's [website](http://www.saifmohammad.com/WebPages/affectintweets.htm).  

If you want to use the script purely a format checker, evaluate your predictions against themselves:

 ```bash
python evaluate.py 1 EI-reg_en_fear_pred.txt EI-reg_en_fear_pred.txt
```

## 2. Weka Baseline System
We have implemented a [WEKA](http://www.cs.waikato.ac.nz/~ml/weka/) package called [AffectiveTweets](https://affectivetweets.cms.waikato.ac.nz/) to be used as a baseline system. The package allows calculating multiple features from a tweet. Installation instructions are given in the project's [webpage](https://affectivetweets.cms.waikato.ac.nz/install/). Make sure to install version 1.0.1 as well as the LibLinear package before running the baselines.  

### 2.1. Data to Arff

We have also implemented the [tweets_to_arff.py](tweets_to_arff.py) script for converting the task data into [arff](http://weka.wikispaces.com/ARFF) format.


### 2.2.  

and the [fix_weka_output.py](fix_weka_output.py) script for converting weka predictions into the official submission format.   




### 2.3. Examples

1. Convert training and target data for the anger emotion into arff format:

 ```bash
python tweets_to_arff.py data/anger-ratings-0to1.train.txt data/anger-ratings-0to1.train.arff
python tweets_to_arff.py data/anger-ratings-0to1.test.target.txt data/anger-ratings-0to1.test.target.arff
```
 If testing data hasn't been provided yet, you can split the training file into training and testing sub-samples. 

2. Train an SVM regression (from LibLinear) on the training data using lexicons, SentiStrength, and word embeddings as features, classify the target tweets, and output the predictions:

 ```bash
java -Xmx4G -cp $HOME/weka-3-8-1/weka.jar weka.Run weka.classifiers.meta.FilteredClassifier -t data/anger-ratings-0to1.train.arff -T data/anger-ratings-0to1.test.target.arff -classifications "weka.classifiers.evaluation.output.prediction.CSV -use-tab -p first-last -file data/anger-pred.csv" -F "weka.filters.MultiFilter -F \"weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector -I 2 -B $HOME/wekafiles/packages/AffectiveTweets/resources/w2v.twitter.edinburgh.100d.csv.gz -S 0 -K 15 -L -O\" -F \"weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector -I 2 -A -D -F -H -J -L -N -P -Q -R -T -U -O\" -F \"weka.filters.unsupervised.attribute.TweetToSentiStrengthFeatureVector -I 2 -U -O\" -F \"weka.filters.unsupervised.attribute.Reorder -R 5-last,4\"" -W weka.classifiers.functions.LibLINEAR -- -S 12 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000 
```

 Make sure that the LibLinear Weka package has been properly installed. 

3. Convert the predictions into the task format:

 ```bash
python fix_weka_output.py data/anger-pred.txt data/anger-pred.txt
 ```
 
4. Evaluate the predictions: 
 
 ```bash
python evaluate.py 1 data/anger-pred.txt data/anger-ratings-0to1.test.gold.txt
 ```
