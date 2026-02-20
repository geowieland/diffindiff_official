#-----------------------------------------------------------------------
# Name:        didtools (diffindiff package)
# Purpose:     Additional tools for Difference-in-Differences Analysis
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     2.1.5
# Last update: 2026-02-20 17:43
# Copyright (c) 2025-2026 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections.abc import Iterable
from statsmodels.formula.api import ols
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import diffindiff.config as config


def check_columns(
    df: pd.DataFrame, 
    columns: list,
    verbose: bool = config.VERBOSE
    ):

    if len(columns) > 0:
        
        if verbose:
            print(f"Checking if column(s) {', '.join(columns)} exist(s) in data frame", end=" ... ")
        
        missing_columns = [col for col in columns if col not in df.columns]
        
        if verbose:
            print("OK")
        
        if missing_columns:
            raise KeyError(f"Data do not contain column(s): {', '.join(missing_columns)}")

def panel_index(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    verbose: bool = config.VERBOSE
    ):

    to_str = []

    if unit_col is not None:
        if data[unit_col].dtype != 'object':
            data[unit_col] = data[unit_col].astype(str)
            to_str.append(unit_col)
    else:
        if verbose:
            print("NOTE: No unit column was stated")

    if time_col is not None:
        if data[time_col].dtype != 'object':
            data[time_col] = data[time_col].astype(str)
            to_str.append(time_col)
    else:
        if verbose:
            print("NOTE: No time column was stated")

    if verbose and len(to_str) > 0:
        print(f"NOTE: The following columns were converted to str: {', '.join(to_str)}.")

    if config.UNIT_TIME_COL not in data.columns:
        
        if unit_col is not None and time_col is not None:

            data[config.UNIT_TIME_COL] = data[unit_col]+config.DELIMITER+data[time_col]
            
            if verbose:
                print(f"NOTE: The following unit-time-index column was created: {config.UNIT_TIME_COL}.")

        else:

            if verbose:
                print("No unit-time-index column was created.")

    return data

def is_balanced(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    other_cols: list = None,
    verbose: bool = config.VERBOSE
    ):

    if verbose:
        print(f"Checking whether panel data is balanced with respect to {3+len(other_cols)} columns", end = " ... ")

    unit_freq = data[unit_col].nunique()
    time_freq = data[time_col].nunique()
    unitxtime = unit_freq*time_freq

    if other_cols is None:
        cols_relevant = [unit_col, time_col, outcome_col]
    else:
        cols_relevant = [unit_col, time_col, outcome_col] + other_cols

    data_relevant = data[cols_relevant]
    
    if unitxtime != len(data_relevant.notna()):
        balanced = False
    else:
        balanced = True
    
    if verbose:
        print("OK")

    if not balanced:
        print("WARNING: Panel data is not balanced.")
        
    return balanced

def is_dummy(
    iterable,
    verbose: bool = config.VERBOSE
    ):

    if not isinstance(iterable, Iterable):
        raise ValueError(f"Stated parameter 'iterable' is not iterable: {iterable}.")
    
    if verbose:
        print("Checking whether iterable is a dummy variable", end = " ... ")

    iterable_as_series = pd.Series(iterable)

    unique_values = set(iterable_as_series.dropna().unique())

    if unique_values == {0, 1}:
        binary = True
    else:
        binary = False
    
    if verbose:
        print("OK")

    return [
        binary,
        unique_values
    ]

def is_binary(
    data: pd.DataFrame,
    treatment_col: str,
    verbose: bool = config.VERBOSE
    ):
    
    if verbose:
        print(f"Checking whether treatment '{treatment_col}' is binary", end = " ... ")
    
    is_binary_check = is_dummy(
        data[treatment_col],
        verbose=False
        )
    binary = is_binary_check[0]
    unique_values = is_binary_check[1]
    
    if unique_values == {0, 1}:
        treatment_format = "Binary"
    elif unique_values == {0}:
        treatment_format = "Constant (no treatment)"
    elif unique_values == {1}:
        treatment_format = "Constant (no control)"
    elif len(unique_values) > 2:
        treatment_format = "Continuous"
    else:
        treatment_format = "Unknown"
    
    if verbose:
        print("OK")

        if not binary:
            print(f"NOTE: treatment column '{treatment_col}' is not binary. Likely treatment format is: {treatment_format}.")
    
    return [
        binary,
        treatment_format
        ]           

def is_missing(
    data: pd.DataFrame,
    drop_missing: bool = True,
    missing_replace_by_zero: bool = False,
    fill_na = 0,
    verbose: bool = config.VERBOSE
    ):

    if verbose:
        print("Checking whether data frame contains missing values", end = " ... ")

    missing_outcome = data.isnull().any()
    missing_outcome_var = any(missing_outcome == True)

    missing_true_vars = []
    if missing_outcome_var:
        missing_true_vars = [name for name, value in missing_outcome.items() if value]        
        
    if verbose:
        print("OK")

    if len(missing_true_vars) > 0:
        print(f"WARNING: Data frame contains columns with missing values: {', '.join(missing_true_vars)}.")

    if drop_missing and not missing_replace_by_zero:
        
        if verbose:
            print("Dropping rows with missing values", end = " ... ")
        
        data = data.dropna(subset = missing_true_vars)
        
        if verbose:
            print("OK")
        
    if missing_replace_by_zero:
        
        if verbose:
            print(f"Replacing missing values with {fill_na}", end = " ... ")
            
        data[missing_true_vars] = data[missing_true_vars].fillna(0)
        
        if verbose:
            print("OK")

    return [
        missing_outcome_var, 
        missing_true_vars, 
        data,
        drop_missing,
        missing_replace_by_zero
        ]

def is_simultaneous(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    pre_post: bool = False,
    verbose: bool = config.VERBOSE
    ):   

    if pre_post:
        
        if verbose:
            print(f"Checking whether treatment '{treatment_col}' is simultaneous or staggered", end = " ... ")
        
        simultaneous = True
    
    else:
        
        data_isnotreatment = is_notreatment(
            data, 
            unit_col, 
            treatment_col,
            verbose=False
            )
        
        if verbose:
            print(f"Checking whether treatment '{treatment_col}' is simultaneous or staggered", end = " ... ")
        
        treatment_group = data_isnotreatment[1]
        data_TG = data[data[unit_col].isin(treatment_group)]

        data_TG_pivot = data_TG.pivot_table (index = time_col, columns = unit_col, values = treatment_col)

        simultaneous = (data_TG_pivot.nunique(axis=1) == 1).all()

    if verbose:
        print("OK")

        if not simultaneous and data_isnotreatment[0]:
            print(f"NOTE: treatment '{treatment_col}' is not simultaneous.")

    if simultaneous and not data_isnotreatment[0]:
        print(f"WARNING: treatment '{treatment_col}' is simultaneous and does not include a {config.NO_TREATMENT_CG_DESCRIPTION}")

    return simultaneous

def is_notreatment(
    data: pd.DataFrame,
    unit_col: str,
    treatment_col: str,
    verbose: bool = config.VERBOSE
    ):

    if verbose:
        print(f"Checking whether treatment '{treatment_col}' includes a {config.NO_TREATMENT_CG_DESCRIPTION}", end = " ... ")

    data_relevant = data[[unit_col, treatment_col]]

    treatment_timepoints = data_relevant.groupby(unit_col).sum(treatment_col)
    treatment_timepoints = treatment_timepoints.reset_index()

    no_treatment = (treatment_timepoints[treatment_col] == 0).any()
    if (treatment_timepoints[treatment_col] == 0).all():
        no_treatment = False

    treatment_group = treatment_timepoints.loc[treatment_timepoints[treatment_col] > 0, unit_col]
    treatment_group = treatment_group.tolist()
    control_group = treatment_timepoints.loc[treatment_timepoints[treatment_col] == 0, unit_col]
    control_group = control_group.tolist()
    
    if verbose:
        print("OK")

        if not no_treatment:
            print(f"NOTE: treatment '{treatment_col}' does not include a {config.NO_TREATMENT_CG_DESCRIPTION}.")

    return [
        no_treatment, 
        treatment_group, 
        control_group
        ]

def treatment_group_col(
    data: pd.DataFrame,
    unit_col: str,
    treatment_col: str,
    create_TG_col: str = "TG",
    verbose: bool = config.VERBOSE
    ):
    
    isnotreatment = is_notreatment(
        data = data,
        unit_col = unit_col,
        treatment_col = treatment_col
        )
    
    if verbose:
        print(f"Creating treatment group column {create_TG_col} for treatment '{treatment_col}'", end = " ... ")
    
    create_TG_col_exists = False
    if create_TG_col in data.columns:
        create_TG_col_exists = True        
        create_TG_col = f"{config.TG_COL}{config.DELIMITER}{treatment_col}"        

    treatment_group = isnotreatment[1]

    data[create_TG_col] = 0
    data.loc[data[unit_col].astype(str).isin(treatment_group), create_TG_col] = 1

    if verbose:
        print("OK")

        if create_TG_col_exists:
            print(f"NOTE: Column {create_TG_col} already exists. Saved treatment group in column {config.TG_COL}{config.DELIMITER}{treatment_col}.")

    return [
        data, 
        isnotreatment[0], 
        create_TG_col
        ]

def treatment_time_col(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    create_TT_col: str = "TT",
    verbose: bool = config.VERBOSE
    ):

    tt = treatment_times(
        data = data,
        unit_col = unit_col,
        time_col = time_col,
        treatment_col = treatment_col,
        verbose = verbose
        )[1]
    
    data[create_TT_col] = 0
    data.loc[data[time_col].isin(tt), create_TT_col] = 1

    return [
        data,
        tt,
        create_TT_col
    ]          

def untreated_units(
    data: pd.DataFrame,
    unit_col: str,
    treatment_col: list,
    verbose: bool = config.VERBOSE
    ):

    if verbose:
        print(f"Identifying treated and untreated units for treatment(s) {', '.join(treatment_col)}", end = " ... ")

    unit_sum = data.groupby(unit_col)[treatment_col].sum().sum(axis=1).reset_index(name="sum")

    units_treated = unit_sum.loc[unit_sum["sum"] > 0, unit_col]
    units_nontreated = unit_sum.loc[unit_sum["sum"] == 0, unit_col]
    no_units_treated = len(units_treated)
    no_units_nontreated = len(units_nontreated)

    if verbose:
        print("OK")

    return [
        no_units_treated,
        no_units_nontreated,
        units_treated,
        units_nontreated
        ]

def is_parallel(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    pre_post: bool = False,
    alpha = 0.05,
    verbose: bool = config.VERBOSE
    ):
 
    modeldata_isnotreatment = is_notreatment(
        data = data,
        unit_col = unit_col,
        treatment_col = treatment_col,
        verbose = False
        )
    
    if verbose:
        print(f"Testing outcome '{outcome_col}' for parallel time trends", end = " ... ")
    
    if pre_post or not modeldata_isnotreatment:
        parallel = "not_tested"
        test_ols_model = None
    
    treatment_group = modeldata_isnotreatment[1]

    if len(data[(data[unit_col].isin(treatment_group)) & (data[treatment_col] == 1)]) > 0:
        
        first_day_of_treatment = min(data[(data[unit_col].isin(treatment_group)) & (data[treatment_col] == 1)][time_col])
        
        data_test = data[data[time_col] < first_day_of_treatment].copy()
        data_test[config.TG_COL] = 0
        data_test.loc[data_test[unit_col].isin(treatment_group), config.TG_COL] = 1
        
        if config.TIME_COUNTER_COL not in data_test.columns:
            data_test = date_counter(
                df = data_test,
                date_col = time_col, 
                new_col = config.TIME_COUNTER_COL,
                verbose = False
                )
        data_test[f"{config.TG_COL}_x_{config.TIME_COL}"] = data_test[config.TG_COL]*data_test[config.TIME_COUNTER_COL]        

        test_ols_model = ols(f'{outcome_col} ~ {config.TG_COL} + {config.TIME_COUNTER_COL} + {config.TG_COL}_x_{config.TIME_COL}', data = data_test).fit()
        coef_TG_x_t_p = test_ols_model.pvalues[f"{config.TG_COL}_x_{config.TIME_COL}"]

        if coef_TG_x_t_p < alpha:
            parallel = False
        else:
            parallel = True
        
    else:
        parallel = "not_tested"
        test_ols_model = None
    
    if verbose:
        print("OK")

    if parallel == "not_tested":
        print("WARNING: Data could not be tested for parallel time trends.")
      
    return [
        parallel, 
        test_ols_model
        ]

def is_prepost(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    verbose: bool = config.VERBOSE
    ):
    
    if verbose:
        print("Checking whether panel data is pre-post or multi-period", end = " ... ")
    
    prepost = (data.groupby(unit_col)[time_col].nunique().le(2).all())
    
    if verbose:
        print("OK")
    
    if verbose:
        if prepost:
            print("NOTE: Panel data is pre-post.")
        else:
            print("NOTE: Panel data is multi-period panel data.")
            
    return prepost

def is_multiple_treatment_period(
    data: pd.DataFrame,
    unit_col: str,
    treatment_col: str,
    verbose: bool = config.VERBOSE
    ):

    if verbose:
        print(f"Checking treatment '{treatment_col}' for multiple treatment periods", end = " ... ")

    unit_treatment_periods = {}
    multiple_treatment_period = False
    units_multiple = 0

    for unit, data_sub in data.groupby(unit_col):
        
        unit_treatment = data_sub[treatment_col]

        groups = (unit_treatment != unit_treatment.shift()).cumsum()

        periods_count = (unit_treatment == 1).groupby(groups).any().sum()

        unit_treatment_periods[unit] = int(periods_count)

    for value in unit_treatment_periods.values():

        if int(value) > 1:

            multiple_treatment_period = True

            units_multiple = units_multiple+1

    if verbose:
        print("OK")
    
        if units_multiple > 0:
            print(f"NOTE: There are {units_multiple} observational units with multiple treatment periods with respect to treatment '{treatment_col}'.")

    return [
        multiple_treatment_period,
        units_multiple,
        unit_treatment_periods
    ]

def date_counter(
    df: pd.DataFrame,
    date_col: str, 
    new_col: str = config.TIME_COUNTER_COL,
    verbose: bool = config.VERBOSE
    ):
    
    if new_col not in df.columns:
               
        if verbose:
            print(f"Building time counter for time column '{date_col}' in new column '{new_col}'", end = " ... ")    
            
        dates = df[date_col].unique()

        date_counter = pd.DataFrame(
            {
                'date': dates,
                new_col: range(1, len(dates) + 1)
                }
            )

        df = df.merge(
            date_counter,
            left_on = date_col,
            right_on = "date"
            )
    
        if verbose:
            print("OK")
    
    else:
        
        if verbose:
            print(f"Time counter column '{new_col}' already exists in data frame.")
        
        if not pd.api.types.is_numeric_dtype(df[new_col]):
            print(f"WARNING: Time counter column '{new_col}' is not numeric.")
    
    return df

def unique(data):

    if data is None or (isinstance(data, (list, np.ndarray, pd.Series, pd.DataFrame)) and len(data) == 0):
        return []
    
    if isinstance(data, pd.DataFrame):
        values = data.values.ravel()

    elif isinstance(data, pd.Series):
        values = data.values.ravel()

    elif isinstance(data, np.ndarray):
        values = data.ravel()

    elif isinstance(data, list):
        values = data

    elif isinstance(data, set):
        values = list(data)

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
    unique_values = list(np.unique(values))
    return unique_values
 
def treatment_times(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    verbose: bool = config.VERBOSE
    ):

    check_columns(
        df = data,
        columns = [
            unit_col, 
            time_col, 
            treatment_col
            ],
        verbose=verbose
        )
    
    is_multiple_treatment_period(
        data = data,
        unit_col = unit_col,
        treatment_col = treatment_col,
        verbose = verbose
        )[0]
    
    if verbose:
        print(f"Identifying treatment times for treatment '{treatment_col}'", end = " ... ")
    
    tt = list(unique(data.loc[data[treatment_col] == 1, time_col]))

    units = unique(data[unit_col])
    
    units_tt = pd.DataFrame(columns = [unit_col, "treatment_min", "treatment_max"])
    
    for unit in units:
        
        data_unit_tt = data[(data[unit_col] == unit) & (data[treatment_col] == 1)]
        
        if data_unit_tt.empty:
            continue
        
        treatment_min = min(data_unit_tt[time_col])
        treatment_max = max(data_unit_tt[time_col])
        
        units_tt = pd.concat(
            [
                units_tt, 
                pd.DataFrame(
                    {
                        unit_col: [unit], 
                        "treatment_min": [treatment_min], 
                        "treatment_max": [treatment_max]
                        }
                    )
                ],
            ignore_index=True
        )
        
    if verbose:
        print("OK")
        
    return [
        units_tt,
        tt
    ]

def model_wrapper(
    y,
    X,
    model_type: str,
    test_size = 0.2,
    train_size = None,
    model_n_estimators = 1000,
    model_max_features = 0.9,
    model_min_samples_split = 2,
    rf_max_depth = None,
    gb_iterations = 100,
    gb_max_depth = 3,
    gb_learning_rate = 0.1,
    knn_n_neighbors = 5,
    svr_kernel = "rbf",
    xgb_learning_rate = 0.1,
    lgbm_learning_rate = 0.1,
    random_state = 71
    ):

    if model_type not in ["ols", "olsbg", "dtbg", "rf", "gb", "knn", "svr", "xgb", "lgbm"]:
        raise ValueError("Please enter a valid model type ('ols', 'olsbg', 'dtbg', 'rf', 'gb', 'knn', 'svr', 'xgb', 'lgbm')")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size = test_size,
        train_size = train_size,
        random_state = random_state
    )
    
    model = None
    y_pred = None

    if model_type == "ols":
        model = LinearRegression()    
    elif model_type == "olsbg":
        model = BaggingRegressor(
            estimator = LinearRegression(),
            n_estimators = model_n_estimators,         
            random_state = random_state
        )         
    elif model_type == "dtbg":
        model = BaggingRegressor(
            estimator = DecisionTreeRegressor(),
            n_estimators = model_n_estimators,         
            random_state = random_state
        )    
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators = model_n_estimators, 
            max_features = model_max_features,
            min_samples_split = model_min_samples_split,
            max_depth = rf_max_depth,
            random_state = random_state
        )
    elif model_type == "gb":
        model = GradientBoostingRegressor(
            learning_rate = gb_learning_rate,
            n_estimators = gb_iterations, 
            max_features = model_max_features,
            min_samples_split = model_min_samples_split,
            max_depth = gb_max_depth,
            random_state = random_state
        )
    elif model_type == "knn":
        model = KNeighborsRegressor(n_neighbors=knn_n_neighbors)
    elif model_type == "svr":
        model = SVR(kernel=svr_kernel)
    elif model_type == "xgb":
        model = XGBRegressor(
            learning_rate = xgb_learning_rate,
            n_estimators = gb_iterations,
            random_state = random_state
        )
    elif model_type == "lgbm":
        model = LGBMRegressor(
            learning_rate = lgbm_learning_rate,
            n_estimators = gb_iterations,
            random_state = random_state
        )
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = fit_metrics(
        observed = y_test, 
        expected = y_pred,        
        remove_nan = True,
        verbose = False
        )    

    return [
        y_pred,
        model,
        metrics
        ]

def fit_metrics(
    observed, 
    expected,
    indep_vars_no: int = None,
    outcome_col: str = None,
    remove_nan: bool = True,
    verbose: bool = False
    ):

    observed_no = len(observed)
    expected_no = len(expected)

    assert observed_no == expected_no, "Error while calculating fit metrics: Observed and expected differ in length"
    
    if not pd.api.types.is_numeric_dtype(observed):
        raise ValueError("Error while calculating fit metrics: Observed column is not numeric")
    if not pd.api.types.is_numeric_dtype(expected):
        raise ValueError("Error while calculating fit metrics: Expected column is not numeric")
    
    if outcome_col is not None:
        outcome_observed_col = f"{outcome_col}{config.DELIMITER}{config.OBSERVED_SUFFIX}"
        outcome_expected_col = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}"
    else:
        outcome_observed_col = f"outcome{config.DELIMITER}{config.OBSERVED_SUFFIX}"
        outcome_expected_col = f"outcome{config.DELIMITER}{config.EXPECTED_SUFFIX}"
    
    if remove_nan:
        
        observed = observed.reset_index(drop=True)
        expected = expected.reset_index(drop=True)

        obs_exp = pd.DataFrame(
            {
                outcome_observed_col: observed, 
                outcome_expected_col: expected
                }
            )
        
        obs_exp_clean = obs_exp.dropna(
            subset=[
                outcome_observed_col, 
                outcome_expected_col
                ]
            )
        
        observed = obs_exp_clean[outcome_observed_col].to_numpy()
        expected = obs_exp_clean[outcome_expected_col].to_numpy()
    
    else:
        
        if np.isnan(observed).any():
            raise ValueError("Error while calculating fit metrics: Vector with observed data contains NaNs and 'remove_nan' is False")
        if np.isnan(expected).any():
            raise ValueError("Error while calculating fit metrics: Vector with expected data contains NaNs and 'remove_nan' is False")
    
    if verbose:
        print("Calculating model fit metrics", end = " ... ")
    
    observations = len(observed)
    
    residuals = np.array(observed)-np.array(expected)
    residuals_sq = residuals**2
    residuals_abs = abs(residuals)
 
    if any(observed == 0):
        
        if verbose:
            print ("NOTE: Vector 'observed' contains values equal to zero. No APE/MAPE calculated.")
        
        APE = np.full_like(observed, np.nan)
        MAPE = np.nan
        
    else:
        
        APE = abs(observed-expected)/observed
        MAPE = float(np.mean(APE))
        
    sAPE = abs(observed-expected)/((abs(observed)+abs(expected))/2)
    
    model_residuals = pd.DataFrame(
        {
            outcome_observed_col: observed,
            outcome_expected_col: expected,
            "residuals": residuals,
            "residuals_sq": residuals_sq,
            "residuals_abs": residuals_abs,
            "APE": APE,
            "APE_SYM": sAPE
            }
        )

    SSR = float(np.sum(residuals_sq))
    SAR = float(np.sum(residuals_abs))    
    observed_mean = float(np.sum(observed)/observed_no)
    SQT = float(np.sum((observed-observed_mean)**2))
    RSQ = float(1-(SSR/SQT))    
    MSE = float(SSR/observed_no)
    RMSE = float(np.sqrt(MSE))
    MAE = float(SAR/observed_no)    
    sMAPE = float(np.mean(sAPE))
    
    if indep_vars_no is not None and isinstance(indep_vars_no, int):
        RSQ_ADJ = (1-(1-RSQ)*((observations-1)/(observations-indep_vars_no-1)))
        
    else:
            
        RSQ_ADJ = np.nan        

    model_fit_metrics = {
        list(config.MODEL_FIT_METRICS.keys())[0]: SSR,
        list(config.MODEL_FIT_METRICS.keys())[1]: SAR,
        list(config.MODEL_FIT_METRICS.keys())[2]: SQT,
        list(config.MODEL_FIT_METRICS.keys())[3]: RSQ,
        list(config.MODEL_FIT_METRICS.keys())[4]: RSQ_ADJ,
        list(config.MODEL_FIT_METRICS.keys())[5]: MSE,
        list(config.MODEL_FIT_METRICS.keys())[6]: RMSE,
        list(config.MODEL_FIT_METRICS.keys())[7]: MAE,
        list(config.MODEL_FIT_METRICS.keys())[8]: MAPE,
        list(config.MODEL_FIT_METRICS.keys())[9]: sMAPE,        
    }    
    
    if verbose:
        print("OK")
    
    if verbose:
        
        if RSQ_ADJ == np.nan:
            print("NOTE: As no number of independent vars was stated, no Adj. R-Squared is calculated.")

        if len(obs_exp_clean) < len(observed) or len(obs_exp_clean) < len(expected):
            print("NOTE: Vectors 'observed' and/or 'expected' contain NaNs which were dropped.")
            
    modelfit_results = [
        model_residuals,
        model_fit_metrics
    ]

    return modelfit_results

def clean_column_name(value):

    value = str(value).upper()
    value = re.sub(r'[^A-Z0-9_]', '_', value) 
    value = re.sub(r'_+', '_', value)        
    
    return value.strip('_')

def clean_treatment_name(treatment_name):
    
    treatment_name = re.sub(r'\W+', "_", treatment_name)
    
    return treatment_name

def replace_prefix(s, prefix, replace):
    
    if s.startswith(prefix):
        return s.replace(prefix, replace, 1)
    
    return s

def bool_to_YN(val):
    
    if isinstance(val, bool):
        return "YES" if val else "NO"    
    else:        
        return val
    
def check_date_format(
    dates: list = None, 
    date_format: str = "%Y-%m-%d"
    ):
    
    if dates is None:
        dates = []
    
    invalid_dates_included = False
    invalid_dates = []
    
    for date in dates:
        try:
            datetime.strptime(date, date_format)
        except (ValueError, TypeError):
            invalid_dates.append(date)
    
    if len(invalid_dates) > 0:
        invalid_dates_included = True
    
    return [
        invalid_dates_included,
        invalid_dates
    ]