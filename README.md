# RPTune 人品调参

A special way to tune parameter with your RP.

## How does it work
1. Use RandomizedSearchCV to get some paraments and their scores.
2. Generate dataset using paraments as features and scores as label.
3. Train simple model and predict the testset.
4. Output the TopK result.

##Comparing RandomizedSearchCV and RPTune
Use digits dataset in *sklearn.datasets.load_digits*.

tool | best score
--- | ---
RandomizedSearchCV  | 0.932
RPTune  | 0.935

You can run this code directly to see the result.

## Enviroment
- Python3 (May be 2 is also supported, but I haven't test it.)
- six
- numpy
- scipy
- pandas
- sklearn

## How to use
```python
from RPTune import RPTune
rpt = RPTune(your_model,param_dist)
rpt.fit(X,y)
```

For more details, see the code below `if __name__ == '__main__':`(line 137).

## Why the name?
Today is April Fool's Day. **Many beautiful girls invited me for dinner. But I refused.**

I sacrificed my precious time to do this thing. 

According to the law of character conservation, this tool must be very powerful.

