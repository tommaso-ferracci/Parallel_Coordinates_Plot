## Introduction
Parallel coordinates plots are a convenient way of visualizing the hyperparameter search for machine learning and deep learning models. While many platforms for model development (see [wandb](https://wandb.ai/site)) already include this kind of visualization, reproducing it in matplotlib for more advanced customization or publishing can be tricky. This package solves the issue by offering compatibility with the major frameworks for hyperparameter tuning. The current version supports `keras_tuner`, `sklearn` and `optuna`.

## Installation
The package is available at [par-coordinates](https://pypi.org/project/par-coordinates/) and can be installed via:
```
pip install par-coordinates
```
## Tutorial
The jupyter notebook `analyses/tutorial.ipynb` contains detailed examples of the package being used:
- random search in `keras_tuner` for a multilayer perceptron
- grid search and random search in `sklearn` for a random forest classifier
- Bayesian optimization in `optuna` for a XGBoost regressor
- exploratory data analysis of the `mtcars` dataset

The workflow is always the same:
```python
# import package
from par_coordinates import get_results
from par_coordinates import plot_par_coordinates
# get results dataframe for specific tuner
results = get_results.sklearn(random_search, "f1")
# display parallel coordinates plot
fig = plot_par_coordinates(results, labels=["maximum tree depth", "minimum samples in leaf node", 
                           "number of trees", "F1 score"], figsize=(8, 4), curves=True, 
                           linewidth=0.8, alpha=0.8, cmap=plt.get_cmap("copper"))
```
resulting in the following visualization (with room for customization):

![](https://github.com/tommaso-ferracci/Parallel_Coordinates_Plot/raw/main/outputs/figures/random_search.png)

For additional information the official documentation is available [here](https://tommaso-ferracci.github.io/Parallel_Coordinates_Plot/index.html).

## License
The package is freely available under MIT license. The code in `src/par_coordinates/plot.py` is based on [pcp](https://github.com/gregornickel/pcp).