# SemEval-2018 Task 1 Affect in Tweets Evaluation Script
Evaluation Script for [SemEval-2018](http://alt.qcri.org/semeval2018/) Task 1: [Affect in Tweets](http://www.saifmohammad.com/WebPages/affectintweets.htm).  


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
* 1 for regression (EI-reg and V-reg tasks)
* 2 for ordinal classification (EI-oc and V-oc tasks)
* 3 for multi-label classification (E-c tasks).



### 1.3. Format
Each input file must have the format specified in the competition's [website](http://www.saifmohammad.com/WebPages/affectintweets.htm).  

If you want to use the script for format checking, evaluate your predictions against themselves:

 ```bash
python evaluate.py 1 EI-reg_en_fear_pred.txt EI-reg_en_fear_pred.txt
```

## 2. Weka Baseline System
We have implemented a [WEKA](http://www.cs.waikato.ac.nz/~ml/weka/) package called [AffectiveTweets](https://affectivetweets.cms.waikato.ac.nz/) to be used as a baseline system. The package allows calculating multiple features from a tweet and can be used together with many machine learning methods implemented in WEKA. Installation instructions are given in the project's [webpage](https://affectivetweets.cms.waikato.ac.nz/install/). Make sure to install the latest version (1.0.1) as well as the LibLinear package before running the examples from below.  

### 2.1. Data to Arff

The [tweets_to_arff.py](tweets_to_arff.py) script allows you to convert the task data into the [arff](http://weka.wikispaces.com/ARFF) WEKA format.

#### Usage

 ```bash
python tweets_to_arff.py <data_type> <input_file> <output file>
```

The parameter <data_type> can take three possible values:
* 1 for regression data (EI-reg, V-reg)
* 2 for ordinal classification data (EC-oc, V-oc)
* 3 for multi-label emotion classification data (E-C).


### 2.2.  Convert Weka Predictions into the Task's Submission Format

The [fix_weka_output.py](fix_weka_output.py) script can be used for converting [weka predictions](https://weka.wikispaces.com/Making+predictions) into the official submission format.   

#### Usage

 ```bash
python fix_weka_output.py <data_type> <original_test_data> <weka_predictions> <output file>
```

The parameter <data_type> can take two possible values:
* 1 for regression data (EI-reg, V-reg)
* 2 for ordinal classification data (EC-oc, V-oc)

Note: the current version of this script can only convert predictions made for the regression and ordinal classification tasks. We will add support for multi-label classification soon.


### 2.3. Examples

#### SVM Regression on EI-reg-En Anger
In this example we will train an SVM regression (from LibLinear) on EI-reg-En-anger-train using unigrams as features, and we will deploy the classifier on the corresponding development set.


1. Convert training and dev sets into arff format:

 ```bash
python tweets_to_arff.py 1 EI-reg-En-anger-train.txt EI-reg-En-anger-train.arff
python tweets_to_arff.py 1 2018-EI-reg-En-anger-dev.txt 2018-EI-reg-En-anger-dev.arff
```


2. Train the classifier using Weka and save the predictions made on the dev set as a csv file:

 ```bash
java -Xmx4G -cp $WEKA_FOLDER/weka.jar weka.Run weka.classifiers.meta.FilteredClassifier -t EI-reg-En-anger-train.arff -T 2018-EI-reg-En-anger-dev.arff -classifications "weka.classifiers.evaluation.output.prediction.CSV -use-tab -p first-last -file EI-reg-En-anger-weka-predictions.csv" -F "weka.filters.MultiFilter -F \"weka.filters.unsupervised.attribute.TweetToSparseFeatureVector -E 5 -D 3 -I 0 -F -M 2 -G 0 -taggerFile $HOME/wekafiles/packages/AffectiveTweets/resources/model.20120919 -wordClustFile $HOME/wekafiles/packages/AffectiveTweets/resources/50mpaths2.txt.gz -Q 1 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler \\\"weka.core.stopwords.Null \\\" -I 2 -U -tokenizer \\\"weka.core.tokenizers.TweetNLPTokenizer \\\"\" -F \"weka.filters.unsupervised.attribute.Reorder -R 5-last,4\"" -W weka.classifiers.functions.LibLINEAR -- -S 12 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000
```

 Make sure that the LibLinear Weka package is properly installed. Note: the snippet from above may look cryptic. Bear in mind that you can configure your classifier from the WEKA GUI and then copy the resulting configuration to the command line.

3. Convert the predictions into the task format:

 ```bash
python fix_weka_output.py 1 2018-EI-reg-En-anger-dev.txt EI-reg-En-anger-weka-predictions.csv EI-reg_en_anger_pred.txt
 ```
 
4. Evaluate the predictions: 
 
 ```bash
python evaluate.py 1 EI-reg_en_anger_pred.txt 2018-EI-reg-En-anger-dev.txt 
 ```
 
 
#### SVM Ordinal Classification on EI-oc-En Anger 
In this example we will use the Weka [OrdinalClassClassifier](http://weka.sourceforge.net/doc.packages/ordinalClassClassifier/weka/classifiers/meta/OrdinalClassClassifier.html) to train and Ordinal Classification model on  EI-oc-En-anger-train using unigrams as features and an SVM classifier (from LibLinear) as the model. We will deploy the classifier on the corresponding development set.
 
1. Convert training and dev sets into arff format:

 ```bash
python ../workspace/SemEval_2018_Task_1_Eval/tweets_to_arff.py 2 EI-oc-En-anger-train.txt EI-oc-En-anger-train.arff
python ../workspace/SemEval_2018_Task_1_Eval/tweets_to_arff.py 2 2018-EI-oc-En-anger-dev.txt 2018-EI-oc-En-anger-dev.arff
```


2. Train the classifier using Weka and save the predictions made on the dev set as a csv file: 
 

 ```bash
java -Xmx4G -cp $WEKA_FOLDER/weka.jar weka.Run weka.classifiers.meta.FilteredClassifier -t EI-oc-En-anger-train.arff -T 2018-EI-oc-En-anger-dev.arff -classifications "weka.classifiers.evaluation.output.prediction.CSV -use-tab -p first-last -file EI-oc-En-anger-weka-predictions.csv" -F "weka.filters.MultiFilter -F \"weka.filters.unsupervised.attribute.TweetToSparseFeatureVector -E 5 -D 3 -I 0 -F -M 0 -G 0 -taggerFile $HOME/wekafiles/packages/AffectiveTweets/resources/model.20120919 -wordClustFile $HOME/wekafiles/packages/AffectiveTweets/resources/50mpaths2.txt.gz -Q 1 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler \\\"weka.core.stopwords.Null \\\" -I 2 -U -tokenizer \\\"weka.core.tokenizers.TweetNLPTokenizer \\\"\" -F \"weka.filters.unsupervised.attribute.Reorder -R 5-last,4\"" -W weka.classifiers.functions.LibLINEAR -- -S 1 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000 
``` 




3. Convert the predictions into the task format:

```bash
python fix_weka_output.py 2 2018-EI-oc-En-anger-dev.txt EI-oc-En-anger-weka-predictions.csv EI-oc_en_anger_pred.txt 
``` 

 
4. Evaluate the predictions: 
 
 ```bash
python evaluate.py 2  EI-oc_en_anger_pred.txt 2018-EI-oc-En-anger-dev.txt 
 ``` 
 
 
 ## TODO
 Examples for making submissions for multi-label classification will be published soon. We will use [MEKA](http://meka.sourceforge.net/) (a Multi-label Extension to WEKA)  for the multi-label data.
