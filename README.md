# DCROD
The code and synthetic datasets of DCROD.  
dcrod.py in code is the implementation of DCROD.  
The algorithm can be used like:
```Python
from dcrod import DCROD

detector = DCROD(n_neighbors=40)
detector.fit(X)
y_outlier_score = detect.decision_scores_
# y_outlier_score is the outlier scores of samples in X, 
# you can use it to calculate AUC score, or detect outliers by a threshold θ
```
The csv files in dataset is the synthetic datasets used in the article.
