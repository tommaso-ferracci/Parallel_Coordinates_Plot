# collection of functions to retrieve results dataframe for the hyperparameter search for various frameworks
import pandas as pd

def keras_tuner(tuner, metric, num_best_trials=1):
    """
    Retrieves results dataframe for the hyperparameter search from a keras_tuner.tuners object. This library
    supports grid search, random search and Bayesian optimization. It is suited for TensorFlow and, through Keras 3.0,
    for PyTorch and JAX as well.

    Arguments:
        tuner (keras_tuner.tuners): tuner after hyperparameter search
        metric (str): name of the metric to retrieve (example: "val_accuracy")
        num_best_trials (int): how many trials to retrieve, in order of best performance (default to 1)

    Returns:
        results (pd.DataFrame): results dataframe for the hyperparameter search
    """
    # get best trials
    trials = tuner.oracle.get_best_trials(num_best_trials)
    # get hyperparameter names and metric name
    columns = list(trials[0].hyperparameters.values.keys())
    columns.append(metric)
    # create empty dataframe
    results = pd.DataFrame(columns=columns)
    # fill it with results for the best trials
    for trial in trials:
        trial_results = list(trial.hyperparameters.values.values())
        trial_results.append(trial.metrics.get_best_value(metric))
        results = pd.concat([results, pd.DataFrame([trial_results], columns=columns)], ignore_index=True)
    return results

def sklearn(tuner, metric):
    """
    Retrieves results dataframe for the hyperparameter search from a GridSearchCV or RandomizedSearchCV object.

    Arguments:
        tuner (sklearn.model_selection.GridSearchCV/RandomizedSearchCV): fitted model after hyperparameter search
        metric (str): name of the metric to retrieve (example: "mean_test_score")

    Returns:
        results (pd.DataFrame): results dataframe for the hyperparameter search
    """
    hyperparameters = tuner.cv_results_["params"]
    scores = tuner.cv_results_["mean_test_score"]
    # add hyperparameters to the dataframe
    results = pd.DataFrame(hyperparameters)
    # add metric column to the dataframe
    results[metric] = scores
    return results

def optuna(study, metric, custom_columns=[]):
    """
    Retrieves results dataframe for the hyperparameter search from a optuna study object. This library offers advanced tools
    for Bayesian optimization and is compatible with all the major machine learning/deep learning frameworks.

    Arguments:
        study (optuna.study): optimized optuna study after hyperparameter search
        metric (str): name of the metric to retrieve (example: "mse")
        custom_columns (list): list of names for custom columns defined by the user (default to [])

    Returns:
        results (pd.DataFrame): results dataframe for the hyperparameter search
    """
    results_df = study.trials_dataframe()
    custom_columns = ["user_attrs_" + n for n in custom_columns]
    columns_to_drop = ["number", "datetime_start", "datetime_complete", "duration", "state"] + custom_columns
    # remove custom columns and trial state informations
    results = results_df.drop(columns_to_drop, axis=1)
    # set the metric as last column
    swap_columns = list(results.columns)
    swap_columns[0], swap_columns[-1] = swap_columns[-1], swap_columns[0]
    results = results[swap_columns]
    # clean column names
    results.columns = [n.split("params_")[-1] for n in results.columns]
    results.columns = list(results.columns[:-1]) + [metric]
    return results 

