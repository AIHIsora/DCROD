# DCROD
The code and synthetic datasets of DCROD.  
dcrod.py is the implementation of DCROD.  
The algorithm can be used like:
```Python
from dcrod import DCROD

detector = DCROD(n_neighbors=40)
detector.fit(X)
y_outlier_score = detector.decision_scores_
# y_outlier_score is the outlier scores of samples in X, 
# you can use it to calculate AUC score, or detect outliers by a threshold θ
```
The csv files in datasets are the synthetic datasets used in the article.

DCROD paper is published in Expert Systems with Applications (ESWA). If you use DCROD in a scientific publication, we would appreciate citations to the following paper:
```
@article{li2022dcrod,
  author  = {Li, Kangsheng and Gao, Xin and Fu, Shiyuan and Diao, Xinping and Ye, Ping and Xue, Bing and Yu, Jiahao and Huang, Zijian},
  title   = {Robust outlier detection based on the changing rate of directed density ratio},
  journal = {Expert Systems with Applications},
  year    = {2022},
  volume  = {207},
  number  = {117988},
  url     = {https://doi.org/10.1016/j.eswa.2022.117988}
}
```
or:
```
Li, K., Gao, X., Fu, S., Diao, X., Ye, P., Xue, B., Yu, J., & Huang, Z. (2022). Robust outlier detection based on the changing rate of directed density ratio. Expert Systems with Applications, 207, 117988.
```
