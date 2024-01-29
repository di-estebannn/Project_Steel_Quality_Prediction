import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import os


def print_results(model, neural_network):
    """
    Print the mean and standard deviation of the mean squared error results
    corresponding to each of the folds (grid search cross-validation) during
    each combination of the model's parameters

    Parameters:
    - model: GridSearchCV object.
    - neural_network (bool): Whether the model is a neural network.
    """
    print('All results:')
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    for mean, std, params in zip(means, stds,params):
        print('Mean = %0.3f and Standard deviation = +/-%0.03f for %r.' % (mean, std*2, params))

    print('Best parameters found:\n', model.best_params_)

    best_model = model.best_estimator_
    
    if (neural_network):
        n_iter = best_model.n_iter_
        print('Number of iterations for convergence:', n_iter)


def train_model(model_name, model, param_grid, X_train, y_train, X_test, y_test, X_val, y_val, results_df, nn = False):
    """
    Trains a machine learning model using grid search cross-validation and
    prints the mean square error of the model when used to predict data from
    the testing and validation data sets. In addition, it saves the results
    in the data frame that is entered by parameter and then it is returned.


    Parameters:
    - model_name (str): Name of the model.
    - model: Machine learning model.
    - param_grid (dict): Dictionary of hyperparameter grids.
    - X_train, y_train: Training data.
    - X_test, y_test: Testing data.
    - X_val, y_val: Validation data.
    - results_df: DataFrame to store results.
    - nn (bool): Whether the model is a neural network.

    Returns:
    - best_model: Best trained model.
    - results_df: Updated DataFrame with results.
    """
    start_time = time.time()
    
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, scoring = 'neg_mean_squared_error', cv = 10)
    grid_search.fit(X_train, y_train)
    print_results(grid_search, nn)

    end_time = time.time()
    training_duration = (end_time - start_time)/60

    best_model = grid_search.best_estimator_
    best_mean = -grid_search.best_score_
    best_std = np.sqrt(grid_search.cv_results_['std_test_score'][grid_search.best_index_])

    y_pred_test = best_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    y_pred_validation = best_model.predict(X_val)
    mse_validation = mean_squared_error(y_val, y_pred_validation)

    print(f'Mean Squared Error on Test dataset (df_testing_data): {mse_test}')
    print(f'Mean Squared Error on Validation dataset (df_validation_data): {mse_validation}')

    results_df.loc[len(results_df)] = {
        'Model': model_name,
        'Mean': best_mean,
        'Standard deviation': best_std,
        'MSE on Testing set': mse_test,
        'MSE on Validation set': mse_validation,
        'Training duration': training_duration,
        'Parameters': grid_search.best_params_
    }
    return best_model, results_df


def learning_curve_model(model, X, y, model_name, path_for_figures, figure_name):
    """
    Plot the learning curve of a machine learning model.

    Parameters:
    - model: Machine learning model.
    - X, y: Data for the learning curve.
    - model_name (str): Name of the model.
    - path_for_figures (str): Path to save the figure.
    - figure_name (str): Name of the saved figure.
    """
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, scoring = 'neg_mean_squared_error', train_sizes = np.linspace(0.1, 0.5, 10))

    train_scores_mean = -np.mean(train_scores, axis = 1)
    test_scores_mean = -np.mean(test_scores, axis = 1)

    plt.figure(figsize = (10, 6))
    plt.plot(train_sizes, train_scores_mean, label = 'Training Error')
    plt.plot(train_sizes, test_scores_mean, label = 'Testing Error')

    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)

    learning_curve_path = os.path.join(path_for_figures, figure_name)
    plt.savefig(learning_curve_path)


def feature_importance(features, importances, model_name, path_for_figures, figure_name):
    """
    Plot the feature importances of a machine learning model.

    Parameters:
    - features: Feature matrix.
    - importances: Feature importances.
    - model_name (str): Name of the model.
    - path_for_figures (str): Path to save the figure.
    - figure_name (str): Name of the saved figure.
    """
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize = (10, 6))
    plt.bar(range(features.shape[1]), importances[indices], align = "center")
    plt.xticks(range(features.shape[1]), indices)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance - {model_name}")

    features_importance_path = os.path.join(path_for_figures, figure_name)
    plt.savefig(features_importance_path)


def save_data(df, data_path, name_file):
    """
    Save results DataFrame to a CSV file.

    Parameters:
    - df: Results DataFrame.
    - data_path (str): Path to save the file.
    - name_file (str): Name of the saved file.
    """
    try:
        df = df.sort_index()
        column_names =  ['Model', 'Mean', 'Standard deviation', 'MSE on Testing set', 'MSE on Validation set', 'Training duration', 'Parameters']
        df.columns = column_names

        df.to_csv(os.path.join(data_path, name_file), index = False)
        print(f"The {name_file} has been saved in {data_path}.\n")

    except Exception as e:
        print(f"Could not save {name_file} in project/data/processed. Error: {str(e)}\n")