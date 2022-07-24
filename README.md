# SMOTE-CDNN

To accelerate future projects using our algorithm and also evaluate SMOTE-CDNN against other resampling algorithms. We develop and open-source Edited CDNN and SMOTE-CDNN as a Python library for performing multi-class hybrid sampling tasks, based on the \texttt{imbalanced-learn} API. 




The repositry includes:
- Native Python implementations of Edited CDNN and SMOTE-CDNN alongside a flexible framework for adapting different distance metrics.
- Examples of using SMOTE-CDNN
- A comparision between SMOTE-CDNN and tradditional resampling algorithms on some sample datasets

Please refer to sample.ipynb for examples.

A sample result will look like this:
```
Testing with ecoli dataset

---------------ecoli dataset------------------
Loading data.....
Done loading data!

algorithms			  fit_time		score_time		test_balanced_accuracy
Origin				    0.058391571		0.001705217		0.769074941
SVMSMOTE			    0.045939541		0.001761913		0.789367681
RandomOverSampler	0.070101786		0.001639652		0.789367681
ADASYN			    	0.044573355		0.001797295		0.798626073
BorderlineSMOTE		0.042854548		0.001593637		0.812884465
SMOTE				      0.104508591		0.001663351		0.816272443
SMOTETomek			  0.048437548		0.001647806		0.816272443
RandomUnderSampler0.026580286		0.0037889		0.835456674
SMOTEENN			    0.045919943		0.001593637		0.86275566
SMOTECDNN			    0.185134459		0.00256443		0.882041374

```
```
