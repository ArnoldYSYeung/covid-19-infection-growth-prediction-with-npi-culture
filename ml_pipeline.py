"""
Functions for machine learning pipeline

Author :            Arnold YS Yeung
Date :              May 10th, 2020
    
"""
import pandas as pd
import numpy as np
import random

from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.externals import joblib

def calculate_prediction_factor(predict_labels, true_labels, remove_threshold=0.005):
    """
    Calculate the prediction factor f = prediction/label. Note that label is the numerator to prevent dividing by 0.
    Prediction is rarely 0.
    Arguments:
        - predict_labels (np.array) :      1D numpy array of predicted labels
        - true_labels (np.array) :         1D numpy array of true labels
        - remove_threshold (float) :       remove labels lower than this (None includes all)
    """
    median_prediction_factor = None
    """
    if remove_threshold is not None:
        zero_indices = np.where(true_labels <= remove_threshold)
        true_labels = np.delete(true_labels, zero_indices)
        predict_labels = np.delete(predict_labels, zero_indices)
    """
    prediction_factors = predict_labels / true_labels 

    # if prediction factor contains 0/0 = Nan
    where_are_NaNs = np.isnan(prediction_factors)
    prediction_factors[where_are_NaNs] = 1

    #prediction_factors[prediction_factors == np.nan] = 1
    
    if np.count_nonzero(np.isnan(prediction_factors)) > 0:
      
      df = {'predict': predict_labels.tolist(), 'true': true_labels.tolist(), 'pf': prediction_factors.tolist()}
      df = pd.DataFrame.from_dict(df)
      df.to_csv('./results/df'+'.csv')
      
    median_prediction_factor =  np.median(prediction_factors)
    
    # convert np.inf to np.nan
    prediction_factors[prediction_factors == np.inf] = np.nan 
    
    return median_prediction_factor, prediction_factors


def select_features(norm_feats, train_labels, feat_names, p_threshold, mi_threshold, 
                    random_seed, both_thresholds=False, threshold_to_use=[]):
    """
    Returns numpy array containing features which meets the requirements.
    Arguments:
        - norm_feats (dict{'train': np.array, 'valid': np.array}):
                                        dictionary containing arrays for features
        - train_labels (pd.Series):     containing labels for training data
        - feat_names (list[str, ]):     names of features
        - p_threshold (float):          p-value threshold
        - mi_threshold (float):         mutual information threshold
        - random_seed (int):            random seed
        - both_thresholds (bool):       whether both threshold needs to be met (true) or only one (false)
        - threshold_to_use(list[str, ]):
                                        thresholds to include ('mutual_info', 'f_score')
    Returns
        Dictionary containing feature arrays with irrelevant features removed
    """
    
    feat_select_df = {}
    
    # compute f-score of each feature
    f_scores, p_values = f_regression(norm_feats['train'], train_labels)
    
    feat_select_df['f_score'] = pd.DataFrame.from_dict({'feat_name': feat_names, 
                                                        'f_score': f_scores, 
                                                        'p_value': p_values})
    feat_select_df['f_score'].plot.bar(x="feat_name", 
                                       y=["f_score", "p_value"], 
                                       secondary_y="p_value")
    
    # compute mutual information of each feature
    mutual_info = mutual_info_regression(norm_feats['train'], 
                                         train_labels, 
                                         random_state=random_seed)
    
    feat_select_df['mutual_info'] = pd.DataFrame.from_dict({'feat_name': feat_names, 
                                                            'mutual_info': mutual_info})
    
    feat_select_df['mutual_info'].plot.bar(x="feat_name", y="mutual_info")


    feats_idx_to_keep_fscore = feat_select_df['f_score'].loc[feat_select_df['f_score']['p_value'] <= p_threshold].index.astype(int).to_list()
    feats_idx_to_keep_mi = feat_select_df['mutual_info'].loc[feat_select_df['mutual_info']['mutual_info'] >= mi_threshold].index.astype(int).to_list()
    
    feats_idx_to_keep = []
    
    # combine thresholds
    if both_thresholds:
        feats_idx_to_keep = list(set(feats_idx_to_keep_fscore) & set(feats_idx_to_keep_mi))
    else:
        if 'f_score' in threshold_to_use:
            feats_idx_to_keep += feats_idx_to_keep_fscore
        if 'mutual_info' in threshold_to_use:
            feats_idx_to_keep += feats_idx_to_keep_mi 
        feats_idx_to_keep = list(set(feats_idx_to_keep))
        
    # remove features from training, validation, and test data
    sel_norm_feats = {}
    
    for key in norm_feats:
        sel_norm_feats[key] = norm_feats[key][:, feats_idx_to_keep]
        
    return sel_norm_feats


def split_by_country(df, n_splits = 10, random_seed=None):
    """
    Split countries into training and validation datasets for cross-validation.
    Arguments:
        - df (pd.DataFrame):            dataframe containing 'CountryName'
        - n_splits (int) :              number of folds to split countries into
        - random_seed (int) :           random seed for shuffling countries
    
    Returns:
        Dictionary of lists, with each dictionary containing a list
            of countries for training and a list of 
            
        {'train': list[list[],] , 'valid': list[list[], ]}
    """

    random.seed(random_seed)
    datasets = {'train': [], 'valid': []}

    if n_splits < 2:
        raise ValueError("We need more splits.")

    ratios = [(n_splits-1)/n_splits, 1/n_splits]
    
    #   count number of countries
    num_countries = df['CountryName'].nunique()
    print("There are "+str(num_countries)+" countries.")
    
    #   split countries into validation and training 
    countries = list(df['CountryName'].unique())
    random.shuffle(countries)
    
    num_valid = int(np.floor(ratios[1]*num_countries))
    num_train = num_countries - num_valid
    
    for start_idx in range(0, num_countries-1, num_valid):
        
        if start_idx == 0:
            valid_countries = countries[start_idx:start_idx+num_valid]
            train_countries = countries[start_idx+num_valid:]
        elif start_idx > num_countries - 2*num_valid:       # last fold
            valid_countries = countries[start_idx:]
            train_countries = countries[0:start_idx]
        else:
            valid_countries = countries[start_idx:start_idx+num_valid]
            train_countries = countries[0:start_idx] + countries[start_idx+num_valid:]
        
        datasets['train'].append(train_countries)
        datasets['valid'].append(valid_countries)
    
    #   remove remainder countries stored in the last fold and share across existing folds
    if len(datasets['valid'][-1]) < len(datasets['valid'][0]):
        for i, remainder_country in enumerate(datasets['valid'][-1]):
            datasets['valid'][i].append(remainder_country)
            datasets['train'][i].remove(remainder_country)
    
        datasets['valid'] = datasets['valid'][:-1]
        datasets['train'] = datasets['train'][:-1]
    
    return datasets

def get_dataframe_by_country(df, country_list):
    
    output_df = df.loc[df['CountryName'].isin(country_list)]
    return output_df


def fit_and_evaluate_model(df, model):
    
    #   separate into cross-validation folds

    feats = sel_norm_feats
    train_features = feats['train']
    valid_features = feats['valid']
    
    #   save best performing models
    top_model_configs = {}
    


def ridge_regression_cv(feats, labels, seed, top_model_configs):
    """
    Runs cross-validation for Ridge Regression.
    Arguments:
        - feats (dict{'train': [np.array, ], 'valid': [np.array, ]}):
                    training and validation features in numpy array format
        - labels (dict{'train': [pd.Series, ], 'valid': [pd.Series, ]}):
                    training and validation labels in Series format
        - seed (int) :                      random seed
        - top_model_configs (dict{}) :      contains the hyperparameters for the best model
    Returns:
        - Dictionary containing hyperparameter values and corresponding
            training and validation errors {hyperparameters: [], 'valid_error': [], 'train_error': []}
        - Updated top_model_configs to include best model from this model type
        - Dictionary containing {'label': np.array, 'predict': np.array} all validation sample
            of the model configuration with the minimum validation error
    
    """
    
    from sklearn.linear_model import Ridge
    
    filename = "./models/lr_reg.pkl"
    
    alphas = list(np.arange(0, 1.25, 0.25))

    results = {'alpha': [], 'valid_error': [], 'train_error': [], 'predict_factor': []}
      
    min_error = None
    best_labels_predicts = None
    
    for alpha in alphas:
        
        mean_valid_error = 0
        mean_train_error = 0
        mean_predict_factor = 0
        
        labels_predicts = {'label': None, 'predict': None, 'p_factor': None}
        
        for fold in range(0, len(feats['train'])):
            
            train_feats = feats['train'][fold]
            valid_feats = feats['valid'][fold]
            train_labels = labels['train'][fold]
            valid_labels = labels['valid'][fold]
                        
            ridge_regressor = Ridge(alpha=alpha, random_state=seed).fit(train_feats, 
                                                                        train_labels)
            
            train_predict = ridge_regressor.predict(train_feats)
            valid_predict = ridge_regressor.predict(valid_feats)
                       
            mean_valid_error += mean_absolute_error(valid_labels, valid_predict)
            mean_train_error += mean_absolute_error(train_labels, train_predict)  
            
            predict_factor, prediction_factors = calculate_prediction_factor(valid_predict, valid_labels.to_numpy())
            mean_predict_factor += predict_factor
            
            if labels_predicts['label'] is None:
                labels_predicts['label'] = valid_labels.to_numpy()
                labels_predicts['predict'] = valid_predict
                labels_predicts['p_factor'] = prediction_factors
            else:
                
                labels_predicts['label'] = np.hstack((labels_predicts['label'], valid_labels.to_numpy()))
                labels_predicts['predict'] = np.hstack((labels_predicts['predict'], valid_predict))
                labels_predicts['p_factor'] = np.hstack((labels_predicts['p_factor'], prediction_factors))
            
        mean_valid_error /= len(feats['train'])
        mean_train_error /= len(feats['valid'])
        mean_predict_factor /= len(feats['valid'])
        
        min_error, has_updated = save_min_error_model(min_error, 
                                                      mean_valid_error, 
                                                      ridge_regressor, 
                                                      filename)
        if has_updated:
            top_model_configs['LinearRegressionReg'] = {'alpha': alpha, 
                                                        'valid_error': mean_valid_error, 
                                                        'train_error': mean_train_error,
                                                        'predict_factor': mean_predict_factor}
            best_labels_predicts = labels_predicts
        
        results['alpha'].append(alpha)
        results['valid_error'].append(mean_valid_error)
        results['train_error'].append(mean_train_error)
        results['predict_factor'].append(mean_predict_factor)
    
    return results, top_model_configs, best_labels_predicts

def decision_tree_regression_cv(feats, labels, seed, top_model_configs):
    """
    Runs cross-validation for Decision Tree Regression.
    Arguments:
        - feats (dict{'train': [np.array, ], 'valid': [np.array, ]}):
                    training and validation features in numpy array format
        - labels (dict{'train': [pd.Series, ], 'valid': [pd.Series, ]}):
                    training and validation labels in Series format
        - seed (int) :                      random seed
        - top_model_configs (dict{}) :      contains the hyperparameters for the best model
    Returns:
        - Dictionary containing hyperparameter values and corresponding
            training and validation errors {hyperparameters: [], 'valid_error': [], 'train_error': []}
        - Updated top_model_configs to include best model from this model type
        - Dictionary containing {'label': np.array, 'predict': np.array} all validation sample
            of the model configuration with the minimum validation error
    
    """
    
    from sklearn.tree import DecisionTreeRegressor
    
    filename = "./models/tree.pkl"
    
    criterion='mse'
    splitter='best'
    depths= [5, 10, 15, 20, 25, 30]
    min_samples_splits = [2, 5, 10]
    min_samples_leaves = [1, 2, 4, 8, 10]
    num_features = ['log2']
    
    results = {'criterion': [], 'depth': [], 'split': [], 'leaf': [], 'features': [], 
               'valid_error': [], 'train_error': [], 'predict_factor': []}

    min_error = None
    best_labels_predicts = None

    for depth in depths:
        for split in min_samples_splits:
            for leaf in min_samples_leaves:
                for n_features in num_features:
                    
                    mean_valid_error = 0
                    mean_train_error = 0
                    mean_predict_factor = 0
                    
                    labels_predicts = {'label': None, 'predict': None, 'p_factor': None}
                    
                    for fold in range(0, len(feats['train'])):
                        
                        train_feats = feats['train'][fold]
                        valid_feats = feats['valid'][fold]
                        train_labels = labels['train'][fold]
                        valid_labels = labels['valid'][fold]
                        
                        tree_regressor = DecisionTreeRegressor(criterion=criterion,
                                                               splitter=splitter,
                                                               max_depth=depth,
                                                               min_samples_split=split,
                                                               min_samples_leaf=leaf,
                                                               max_features=n_features,
                                                               random_state=seed)
                        
                        tree_regressor = tree_regressor.fit(train_feats, train_labels)
                        
                        train_predict = tree_regressor.predict(train_feats)
                        valid_predict = tree_regressor.predict(valid_feats)
                                   
                        mean_valid_error += mean_absolute_error(valid_labels, valid_predict)
                        mean_train_error += mean_absolute_error(train_labels, train_predict)
                        
                        predict_factor, prediction_factors = calculate_prediction_factor(valid_predict, valid_labels.to_numpy())
                        mean_predict_factor += predict_factor
                        
                        if labels_predicts['label'] is None:
                            labels_predicts['label'] = valid_labels.to_numpy()
                            labels_predicts['predict'] = valid_predict
                            labels_predicts['p_factor'] = prediction_factors
                        else:
                            
                            labels_predicts['label'] = np.hstack((labels_predicts['label'], valid_labels.to_numpy()))
                            labels_predicts['predict'] = np.hstack((labels_predicts['predict'], valid_predict))  
                            labels_predicts['p_factor'] = np.hstack((labels_predicts['p_factor'], prediction_factors))            
                            
                    mean_valid_error /= len(feats['train'])
                    mean_train_error /= len(feats['valid'])    
                    mean_predict_factor /= len(feats['valid'])
                                        
                    min_error, has_updated = save_min_error_model(min_error, 
                                                                  mean_valid_error, 
                                                                  tree_regressor, 
                                                                  filename)
                    if has_updated:
                        top_model_configs['DecisionTree'] = {'depth': depth, 
                                                             'split': split, 
                                                             'leaf': leaf, 
                                                             'n_features': n_features, 
                                                             'valid_error': mean_valid_error,
                                                             'train_error': mean_train_error,
                                                             'predict_factor': mean_predict_factor}
                        best_labels_predicts = labels_predicts
                    
                    results['criterion'].append(criterion)
                    results['depth'].append(depth)
                    results['split'].append(split)
                    results['leaf'].append(leaf)
                    results['features'].append(n_features)
                    results['valid_error'].append(mean_valid_error)
                    results['train_error'].append(mean_train_error)
                    results['predict_factor'].append(mean_predict_factor)
            
    return results, top_model_configs, best_labels_predicts

def sv_regression_cv(feats, labels, seed, top_model_configs):
    """
    Runs cross-validation for Support Vector Regression.
    Arguments:
        - feats (dict{'train': [np.array, ], 'valid': [np.array, ]}):
                    training and validation features in numpy array format
        - labels (dict{'train': [pd.Series, ], 'valid': [pd.Series, ]}):
                    training and validation labels in Series format
        - seed (int) :                      random seed
        - top_model_configs (dict{}) :      contains the hyperparameters for the best model
    Returns:
        - Dictionary containing hyperparameter values and corresponding
            training and validation errors {hyperparameters: [], 'valid_error': [], 'train_error': []}
        - Updated top_model_configs to include best model from this model type
        - Dictionary containing {'label': np.array, 'predict': np.array} all validation sample
            of the model configuration with the minimum validation error
    
    """
    
    from sklearn.svm import SVR
    
    filename = "./models/svr.pkl"
    
    #kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernels = ['linear', 'rbf', 'sigmoid']
    epsilons = [0, 0.1, 0.2, 0.5]
    
    results = {'kernel': [], 'epsilon': [],
               'valid_error': [], 'train_error': [],
               'predict_factor': []}

    min_error = None
    best_labels_predicts = None

    for kernel in kernels:
        
        for epsilon in epsilons:
                    
            print("Kernel:", kernel, "  Epsilon:", epsilon)
            
            mean_valid_error = 0
            mean_train_error = 0
            mean_predict_factor = 0
            
            labels_predicts = {'label': None, 'predict': None, 'p_factor': None}
            
            for fold in range(0, len(feats['train'])):
                
                if not isinstance(fold, int):
                    print("Fold" + str(fold) + str(type(fold)))
                
                train_feats = feats['train'][fold]
                valid_feats = feats['valid'][fold]
                train_labels = labels['train'][fold]
                valid_labels = labels['valid'][fold]
                
                sv_regressor = SVR(kernel=kernel, epsilon=epsilon)
                
                sv_regressor = sv_regressor.fit(train_feats, train_labels)
                
                train_predict = sv_regressor.predict(train_feats)
                valid_predict = sv_regressor.predict(valid_feats)
                           
                mean_valid_error += mean_absolute_error(valid_labels, valid_predict)
                mean_train_error += mean_absolute_error(train_labels, train_predict)
                
                predict_factor, prediction_factors = calculate_prediction_factor(valid_predict, valid_labels.to_numpy())
                mean_predict_factor += predict_factor
                
                if labels_predicts['label'] is None:
                    labels_predicts['label'] = valid_labels.to_numpy()
                    labels_predicts['predict'] = valid_predict
                    labels_predicts['p_factor'] = prediction_factors
                else:
                    
                    labels_predicts['label'] = np.hstack((labels_predicts['label'], valid_labels.to_numpy()))
                    labels_predicts['predict'] = np.hstack((labels_predicts['predict'], valid_predict))
                    labels_predicts['p_factor'] = np.hstack((labels_predicts['p_factor'], prediction_factors))
                                
            mean_valid_error /= len(feats['train'])
            mean_train_error /= len(feats['valid'])
            mean_predict_factor /= len(feats['valid'])
                                
            min_error, has_updated = save_min_error_model(min_error, 
                                                          mean_valid_error, 
                                                          sv_regressor, 
                                                          filename)
            if has_updated:
                top_model_configs['SVR'] = {'kernel': kernel, 
                                            'epsilon': epsilon, 
                                            'valid_error': mean_valid_error, 
                                            'train_error': mean_train_error,
                                            'predict_factor': mean_predict_factor}
                best_labels_predicts = labels_predicts
            
            results['kernel'].append(kernel)
            results['epsilon'].append(epsilon)
            results['valid_error'].append(mean_valid_error)
            results['train_error'].append(mean_train_error)
            results['predict_factor'].append(mean_predict_factor)
            
    return results, top_model_configs, best_labels_predicts

def random_forest_regression_cv(feats, labels, seed, top_model_configs):
    """
    Runs cross-validation for Random Forest Regression.
    Arguments:
        - feats (dict{'train': [np.array, ], 'valid': [np.array, ]}):
                    training and validation features in numpy array format
        - labels (dict{'train': [pd.Series, ], 'valid': [pd.Series, ]}):
                    training and validation labels in Series format
        - seed (int) :                      random seed
        - top_model_configs (dict{}) :      contains the hyperparameters for the best model
    Returns:
        - Dictionary containing hyperparameter values and corresponding
            training and validation errors {hyperparameters: [], 'valid_error': [], 'train_error': []}
        - Updated top_model_configs to include best model from this model type
        - Dictionary containing {'label': np.array, 'predict': np.array} all validation sample
            of the model configuration with the minimum validation error
    
    """
    
    from sklearn.ensemble import RandomForestRegressor
    
    filename = "./models/rf.pkl"
    
    n_estimators = [3, 5, 10, 15, 20, 30, 50, 75, 100, 125, 150]
    criterion = 'mse' 
    depths= [5, 10, 15, 20, 25, 30]
    min_samples_splits = [2, 5, 10]
    min_samples_leaves = [1, 2, 4, 8, 10]
    num_features = ['log2']
    
    results = {'criterion': [], 
               'estimators': [], 
               'depth': [], 
               'split': [], 
               'leaf': [], 
               'features': [], 
               'valid_error': [], 
               'train_error': [],
               'predict_factor': []}
    
    min_error = None
    best_labels_predicts = None
    
    print("This will take several minutes.  Please be patient.")

    print("Num Estimators: ", end="")
    for num_estimator in n_estimators:
        
        print(num_estimator, end=" ")
        for depth in depths:
            for split in min_samples_splits:
                for leaf in min_samples_leaves:
                    for n_features in num_features:
                        
                        mean_valid_error = 0
                        mean_train_error = 0
                        mean_predict_factor = 0
                        
                        labels_predicts = {'label': None, 'predict': None, 'p_factor':None}
                                        
                        for fold in range(0, len(feats['train'])):
                                                        
                            train_feats = feats['train'][fold]
                            valid_feats = feats['valid'][fold]
                            train_labels = labels['train'][fold]
                            valid_labels = labels['valid'][fold]
                            
                            rf_regressor = RandomForestRegressor(n_estimators=num_estimator, 
                                                                 criterion=criterion, 
                                                                 max_depth=depth, 
                                                                 min_samples_split=split, 
                                                                 min_samples_leaf=leaf,
                                                                 max_features=n_features, 
                                                                 random_state=seed)
                            
                            rf_regressor = rf_regressor.fit(train_feats, train_labels)
                            
                            train_predict = rf_regressor.predict(train_feats)
                            valid_predict = rf_regressor.predict(valid_feats)
                                       
                            mean_valid_error += mean_absolute_error(valid_labels, valid_predict)
                            mean_train_error += mean_absolute_error(train_labels, train_predict)
                            
                            predict_factor, prediction_factors = calculate_prediction_factor(valid_predict, valid_labels.to_numpy())
                            mean_predict_factor += predict_factor
                        
                            if labels_predicts['label'] is None:
                                labels_predicts['label'] = valid_labels.to_numpy()
                                labels_predicts['predict'] = valid_predict
                                labels_predicts['p_factor'] = prediction_factors
                            else:
                                
                                labels_predicts['label'] = np.hstack((labels_predicts['label'], valid_labels.to_numpy()))
                                labels_predicts['predict'] = np.hstack((labels_predicts['predict'], valid_predict))  
                                labels_predicts['p_factor'] = np.hstack((labels_predicts['p_factor'], prediction_factors))            
                                
                        mean_valid_error /= len(feats['train'])
                        mean_train_error /= len(feats['valid'])    
                        mean_predict_factor /= len(feats['valid'])
                                            
                        min_error, has_updated = save_min_error_model(min_error, 
                                                                      mean_valid_error, 
                                                                      rf_regressor, 
                                                                      filename)
                        if has_updated:
                            top_model_configs['RandomForest'] = {'estimators': num_estimator, 
                                                                 'depth': depth, 
                                                                 'split': split, 
                                                                 'leaf': leaf, 
                                                                 'n_features': n_features, 
                                                                 'valid_error': mean_valid_error, 
                                                                 'train_error': mean_train_error,
                                                                 'predict_factor': mean_predict_factor}
                            best_labels_predicts = labels_predicts

                        results['criterion'].append(criterion)
                        results['estimators'].append(num_estimator)
                        results['depth'].append(depth)
                        results['split'].append(split)
                        results['leaf'].append(leaf)
                        results['features'].append(n_features)
                        results['valid_error'].append(mean_valid_error)
                        results['train_error'].append(mean_train_error)
                        results['predict_factor'].append(mean_predict_factor)
    
    return results, top_model_configs, best_labels_predicts

def adaboost_regression_cv(feats, labels, seed, top_model_configs):
    """
    Runs cross-validation for AdaBoost Regression (for decision trees).
    Arguments:
        - feats (dict{'train': [np.array, ], 'valid': [np.array, ]}):
                    training and validation features in numpy array format
        - labels (dict{'train': [pd.Series, ], 'valid': [pd.Series, ]}):
                    training and validation labels in Series format
        - seed (int) :                      random seed
        - top_model_configs (dict{}) :      contains the hyperparameters for the best model
    Returns:
        - Dictionary containing hyperparameter values and corresponding
            training and validation errors {hyperparameters: [], 'valid_error': [], 'train_error': []}
        - Updated top_model_configs to include best model from this model type
        - Dictionary containing {'label': np.array, 'predict': np.array} all validation sample
            of the model configuration with the minimum validation error
    
    """
    
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor
    
    filename = "./models/adaboost.pkl"
    
    weak_learner = DecisionTreeRegressor(max_depth=3)
    n_estimators = [3, 5, 10, 15, 20, 30, 50, 75, 100, 125, 150]
    learning_rates = [0.1, 0.5, 1]
    loss = 'linear'
    
    results = {'loss': [], 
               'estimators': [], 
               'learning_rate': [], 
               'valid_error': [], 
               'train_error': [],
               'predict_factor': []}
    
    min_error = None
    best_labels_predicts = None
    
    print("This will take several minutes.  Please be patient.")

    print("Num Estimators: ", end="")
    for num_estimator in n_estimators:
        
        print(num_estimator, end=" ")
        for learning_rate in learning_rates:
                        
            mean_valid_error = 0
            mean_train_error = 0
            mean_predict_factor = 0
            
            labels_predicts = {'label': None, 'predict': None, 'p_factor':None}
                            
            for fold in range(0, len(feats['train'])):
                                            
                train_feats = feats['train'][fold]
                valid_feats = feats['valid'][fold]
                train_labels = labels['train'][fold]
                valid_labels = labels['valid'][fold]
                
                adaboost_regressor = AdaBoostRegressor(base_estimator=weak_learner,
                                                       n_estimators=num_estimator, 
                                                       loss=loss, 
                                                       learning_rate=learning_rate, 
                                                       random_state=seed)
                
                adaboost_regressor = adaboost_regressor.fit(train_feats, train_labels)
                
                train_predict = adaboost_regressor.predict(train_feats)
                valid_predict = adaboost_regressor.predict(valid_feats)
                           
                mean_valid_error += mean_absolute_error(valid_labels, valid_predict)
                mean_train_error += mean_absolute_error(train_labels, train_predict)
                
                predict_factor, prediction_factors = calculate_prediction_factor(valid_predict, valid_labels.to_numpy())
                mean_predict_factor += predict_factor
            
                if labels_predicts['label'] is None:
                    labels_predicts['label'] = valid_labels.to_numpy()
                    labels_predicts['predict'] = valid_predict
                    labels_predicts['p_factor'] = prediction_factors
                else:
                    
                    labels_predicts['label'] = np.hstack((labels_predicts['label'], valid_labels.to_numpy()))
                    labels_predicts['predict'] = np.hstack((labels_predicts['predict'], valid_predict))  
                    labels_predicts['p_factor'] = np.hstack((labels_predicts['p_factor'], prediction_factors))            
                    
            mean_valid_error /= len(feats['train'])
            mean_train_error /= len(feats['valid'])    
            mean_predict_factor /= len(feats['valid'])
                                
            min_error, has_updated = save_min_error_model(min_error, 
                                                          mean_valid_error, 
                                                          adaboost_regressor, 
                                                          filename)
            if has_updated:
                top_model_configs['AdaBoost'] = {'estimators': num_estimator, 
                                                 'loss': loss, 
                                                 'learning_rate': learning_rate,  
                                                 'valid_error': mean_valid_error, 
                                                 'train_error': mean_train_error,
                                                 'predict_factor': mean_predict_factor}
                best_labels_predicts = labels_predicts

            results['loss'].append(loss)
            results['estimators'].append(num_estimator)
            results['learning_rate'].append(learning_rate)
            results['valid_error'].append(mean_valid_error)
            results['train_error'].append(mean_train_error)
            results['predict_factor'].append(mean_predict_factor)
    
    return results, top_model_configs, best_labels_predicts

def mlp_regression_cv(feats, labels, seed, top_model_configs):
    """
    Runs cross-validation for Multi-Layer Perceptron.
    Arguments:
        - feats (dict{'train': [np.array, ], 'valid': [np.array, ]}):
                    training and validation features in numpy array format
        - labels (dict{'train': [pd.Series, ], 'valid': [pd.Series, ]}):
                    training and validation labels in Series format
        - seed (int) :                      random seed
        - top_model_configs (dict{}) :      contains the hyperparameters for the best model
    Returns:
        - Dictionary containing hyperparameter values and corresponding
            training and validation errors {hyperparameters: [], 'valid_error': [], 'train_error': []}
        - Updated top_model_configs to include best model from this model type
        - Dictionary containing {'label': np.array, 'predict': np.array} all validation sample
            of the model configuration with the minimum validation error
    
    """
    
    from sklearn.neural_network import MLPRegressor
    
    filename = "./models/mlp.pkl"
    
    layer_sets = [(40, ), (80), (100, ), (100, 100), (300, 300)]
    activations = ['identity', 'relu']
    alphas = [0, 0.0001, 0.0005]
    learning_rates = [0.0001, 0.0005, 0.001, 0.002]
    max_iter = 2000
        
    results = {'layers': [], 'activation': [], 'alpha': [], 
               'learning_rate': [], 'max_iter': [], 'valid_error': [], 
               'train_error': [], 'predict_factor': []}

    min_error = None
    best_labels_predicts = None

    for layers in layer_sets:
        print(layers)
        for learning_rate in learning_rates:
            print(learning_rate)
            for activation in activations:
                for alpha in alphas:
                        
                    mean_valid_error = 0
                    mean_train_error = 0
                    mean_predict_factor = 0
                    
                    labels_predicts = {'label': None, 'predict': None, 'p_factor': None}
                    
                    for fold in range(0, len(feats['train'])):
                        
                        train_feats = feats['train'][fold]
                        valid_feats = feats['valid'][fold]
                        train_labels = labels['train'][fold]
                        valid_labels = labels['valid'][fold]
                        
                        mlp_regressor = MLPRegressor(hidden_layer_sizes=layers, 
                                                     activation=activation, 
                                                     alpha=alpha, 
                                                     learning_rate_init=learning_rate,
                                                     max_iter=max_iter, 
                                                     random_state=seed)
                            
                            
                        mlp_regressor = mlp_regressor.fit(train_feats, train_labels)
                    
                        train_predict = mlp_regressor.predict(train_feats)
                        valid_predict = mlp_regressor.predict(valid_feats)
                                   
                        mean_valid_error += mean_absolute_error(valid_labels, valid_predict)
                        mean_train_error += mean_absolute_error(train_labels, train_predict)
                        
                        predict_factor, prediction_factors = calculate_prediction_factor(valid_predict, valid_labels.to_numpy())
                        mean_predict_factor += predict_factor
                        
                        if labels_predicts['label'] is None:
                            labels_predicts['label'] = valid_labels.to_numpy()
                            labels_predicts['predict'] = valid_predict
                            labels_predicts['p_factor'] = prediction_factors
                        else:
                            
                            labels_predicts['label'] = np.hstack((labels_predicts['label'], valid_labels.to_numpy()))
                            labels_predicts['predict'] = np.hstack((labels_predicts['predict'], valid_predict))  
                            labels_predicts['p_factor'] = np.hstack((labels_predicts['p_factor'], prediction_factors))            
            
                            
                    mean_valid_error /= len(feats['train'])
                    mean_train_error /= len(feats['valid'])
                    mean_predict_factor /= len(feats['valid'])
                                        
                    min_error, has_updated = save_min_error_model(min_error, 
                                                                  mean_valid_error, 
                                                                  mlp_regressor, 
                                                                  filename)
                    if has_updated:
                        top_model_configs['NeuralNetwork'] = {'layers': layers, 
                                                              'lr': learning_rate, 
                                                              'activation': activation, 
                                                              'alpha': alpha, 
                                                              'valid_error': mean_valid_error, 
                                                              'train_error': mean_train_error,
                                                              'predict_factor': mean_predict_factor}
    
                        best_labels_predicts = labels_predicts
                    
                    results['layers'].append(layers)
                    results['activation'].append(activation)
                    results['learning_rate'].append(learning_rate)
                    results['alpha'].append(alpha)
                    results['max_iter'].append(max_iter)
                    results['valid_error'].append(mean_valid_error)
                    results['train_error'].append(mean_train_error)
                    results['predict_factor'].append(mean_predict_factor)
            
    return results, top_model_configs, best_labels_predicts
                    

def save_min_error_model(min_error, valid_error, model, filename):
    """
    Update (if necessary) the minimum validation error and save the new model if 
    updated.
    Arguments:
        - min_error (float) :               current minimum validation error
        - valid_error (float) :             new validation error to compare to
        - model (sklearn model) :           model to save
        - filename (str) :                  name of file to save model to
    Returns:
        - Updated (if necessary) minimum error (float)
        - Whether the minimum error has been updated or not (bool)
    
    """
        
    has_updated = False
    
    # update best performing hyperparamters
    if min_error is None:
        min_error = valid_error
        _ = joblib.dump(model, filename)
        has_updated = True
    else:
        if valid_error < min_error:
            min_error = valid_error
            _ = joblib.dump(model, filename)
            has_updated = True
        
    return min_error, has_updated

def cross_validate_by_country(df, n_splits, random_seed, p_value=0.01, mi=0.10, 
                              both_thresholds=False, threshold_to_use=[], features_to_exclude=[], 
                              label_name='ConfirmedGrowth'):
    """
    Create cross-validation datasets (features and labels), splitting by country.
    """

    features = {'train': [], 'valid': []}
    labels = {'train': [], 'valid': []}
    
    countries_split = split_by_country(df, n_splits, random_seed)
    
    n_folds = len(countries_split['train'])
    
    print("Num Folds:",  n_folds)
    
    for fold in range(0, n_folds):
        
        print("Fold: "+str(fold))
        
        train_countries = countries_split['train'][fold]
        valid_countries = countries_split['valid'][fold]
        
        #   check for countries in both training and validation dataset
        if bool(set(train_countries) & set(valid_countries)):
            raise ValueError("Training and Validation sets share same countries.")
        
        #   separate into training and validation datasets
        train_df = get_dataframe_by_country(df, train_countries)
        valid_df = get_dataframe_by_country(df, valid_countries)
        
        #   separate features from labels
        markers = ['CountryName', 'CountryCode', 'Date', 'rn', 'Confirmed', 'Daily']
        label_names = [label_name]
            
        markers = [marker for marker in markers if marker in train_df.columns]
        label_names = [label_name for label_name in label_names if label_name in train_df.columns]
        features_to_exclude = [feature for feature in features_to_exclude if feature in train_df.columns]

        train_feats = train_df.drop(columns=label_names+markers+features_to_exclude)
        train_labels = train_df[label_names]
        
        valid_feats = valid_df.drop(columns=label_names+markers+features_to_exclude)
        valid_labels = valid_df[label_names]
        
        feat_names = list(train_feats.columns)
        
        #   normalize features
        scaler = MinMaxScaler()

        norm_feats = {'train': None, 'valid': None}
        
        scaler.fit(train_feats)
        norm_feats['train'] = scaler.transform(train_feats)
        norm_feats['valid'] = scaler.transform(valid_feats)
        
        #   feature selection and extraction
        sel_norm_feats = select_features(norm_feats, train_labels[label_name],
                                         feat_names, p_threshold=p_value,
                                         mi_threshold=mi, both_thresholds=both_thresholds,
                                         threshold_to_use=threshold_to_use, random_seed=random_seed)
                        
        #   add to dictionaries
        features['train'].append(sel_norm_feats['train'])
        labels['train'].append(train_labels[label_name])
        features['valid'].append(sel_norm_feats['valid'])
        labels['valid'].append(valid_labels[label_name])
    
    return features, labels

def main():
    
    model_filenames = {'LinearRegression': "./models/lr.pkl",
             'LinearRegressionReg': "./models/lr_reg.pkl",
             'DecisionTree': "./models/tree.pkl",
             'BagLinearRegression': "./models/lr_bag.pkl",
             'RandomForest': "./models/rf.pkl",
             'SVR': "./models/svr.pkl",
             'NeuralNetwork': "./models/nn.pkl",
             'AdaBoost': "./models/adaboost.pkl"
             }