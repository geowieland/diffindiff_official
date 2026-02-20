#-----------------------------------------------------------------------
# Name:        didanalysis_helper (diffindiff package)
# Purpose:     Helper functions for didanalysis module
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.6
# Last update: 2025-02-20 17:38
# Copyright (c) 2025-2026 Thomas Wieland
#-----------------------------------------------------------------------

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy.highlevel import dmatrices
import numpy as np
from datetime import datetime
import diffindiff.didtools as tools
import diffindiff.config as config


def create_fixed_effects(
    data: pd.DataFrame,
    col: str,
    type: str = "unit",
    drop_first: bool = False,
    verbose: bool = config.VERBOSE
    ):
     
    if type not in config.FE_TYPES:
        raise ValueError(f"Parameter 'type' must be one of the following: {', '.join(config.FE_TYPES)}")

    for key, value in config.EFFECTS_TYPES["FE"]["types"].items():

        if value["FE"] == type:
            
            prefix = value["dummy_prefix"]
            
            if verbose:
                print(f"Creating {value['description']} for column '{col}' with prefix '{prefix}{config.DELIMITER}'", end = " ... ")
    
    dummy_unit_original = list(data[col].astype(str).unique())
    
    dummy_unit_vars = [f"{prefix}{config.DELIMITER}{tools.clean_column_name(val)}" for val in dummy_unit_original]    
    
    dummies = pd.DataFrame(
        pd.get_dummies(
            data = data[col].astype(str),
            dtype = int,
            prefix = prefix,
            prefix_sep = config.DELIMITER,
            drop_first = drop_first
            )
        )

    dummies.columns = [tools.clean_column_name(col) for col in dummies.columns]

    data = pd.concat([data, dummies], axis=1)

    dummies_join = " + ".join(dummies.columns)

    if verbose:
        print("OK")

    return [
        data, 
        dummies_join, 
        dummy_unit_vars,
        dummy_unit_original
        ]

def create_specific_time_trends(    
    data: pd.DataFrame,
    time_col: str,
    FE_vars: list,
    type: str = "ITT",
    verbose: bool = config.VERBOSE
    ):
    
    if type not in config.TIME_TRENDS_TYPES:
        raise ValueError(f"Parameter 'type' must be one of the following: {', '.join(config.TIME_TRENDS_TYPES)}")
    
    tools.check_columns(
        df = data,
        columns = FE_vars,
        verbose = verbose
        )
    
    data = tools.date_counter(
        data,
        time_col,
        new_col=config.TIME_COUNTER_COL,
        verbose = verbose
        )
    
    if verbose:
        print(f"Creating {config.EFFECTS_TYPES[type]['description']} for {len(FE_vars)} entities with suffix '{config.DELIMITER}{config.TIME_COL}'", end = " ... ")
    
    id_x_time = pd.DataFrame()
    
    for col in FE_vars:
        
        if col in data.columns:
            
            id_x_time[col] = data[col] * data[config.TIME_COUNTER_COL]
            new_col_name = f"{col}{config.DELIMITER_INTERACT}{config.TIME_COL}"
            id_x_time = id_x_time.rename(columns={col: new_col_name})
            
    data = pd.concat([data, id_x_time], axis = 1)
    
    time_trend_vars = id_x_time.columns
    id_x_time_join = " + ".join(time_trend_vars)
    
    if verbose:
        print("OK")
    
    return [
        data, 
        id_x_time_join, 
        time_trend_vars
        ]

def create_specific_treatment_effects(
    data: pd.DataFrame,
    treatment_col: list,
    FE_vars: list,
    type: str = "ITE",
    verbose: bool = config.VERBOSE
    ):
    
    if type not in config.SPECIFIC_EFFFECTS_TYPES:
        raise ValueError(f"Parameter 'type' must be one of the following: {', '.join(config.SPECIFIC_EFFFECTS_TYPES)}")
    
    tools.check_columns(
        df = data,
        columns = FE_vars,
        verbose = verbose
        )   
    
    if verbose:
        print(f"Creating {config.EFFECTS_TYPES[type]['description']} for {len(FE_vars)} entities and {len(treatment_col)} treatments", end = " ... ")
        
    id_x_treatment = pd.DataFrame()
            
    for col in FE_vars:
        
        if col in data.columns:
        
            for treatment in treatment_col:
                                  
                id_x_treatment[col] = data[col] * data[treatment]
                new_col_name = f"{treatment}{config.DELIMITER}{col}"
                id_x_treatment = id_x_treatment.rename(columns={col: new_col_name})
                
                if id_x_treatment[new_col_name].sum() == 0:
                    id_x_treatment = id_x_treatment.drop(columns=[new_col_name])                    
                
    data = pd.concat([data, id_x_treatment], axis = 1)
    
    id_treatment_vars = id_x_treatment.columns
    id_x_treatment_join = ' + '.join(id_treatment_vars)
    
    if verbose:
        print("OK")

    return [
        data, 
        id_x_treatment_join, 
        id_treatment_vars
        ]
        
def create_spillover(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: list,
    TT_col: str = None,
    spillover_treatment: list = [],
    spillover_units: list = [],
    verbose: bool = config.VERBOSE
    ):

    if spillover_treatment is None or spillover_treatment == []:
        raise ValueError("Parameter 'spillover_treatment' does not contain any treatment")
    if spillover_units is None or treatment_col == []:
        raise ValueError("Parameter 'spillover_units' does not contain any treatment")

    if verbose:
        print(f"Creating spillover variables for treatment(s) {', '.join(treatment_col)} and {len(spillover_units)} units", end = " ... ")

    spillover_unit_vars = []
    spillover_treatment_vars = []

    for treatment in treatment_col:

        if TT_col is None:
            
            TT_col = config.TT_COL

            data = tools.treatment_time_col(
                data = data,
                unit_col = unit_col,
                time_col = time_col,
                treatment_col = treatment,
                create_TT_col = TT_col,
                verbose = verbose
                )[0]          

        sp_unit_col = f"{config.SPILLOVER_UNIT_PREFIX}{config.DELIMITER}{treatment}"
        sp_treatment_col = f"{config.SPILLOVER_PREFIX}{config.DELIMITER}{treatment}"

        data[sp_unit_col] = 0
        data[sp_treatment_col] = 0

        spillover_unit_vars.append(sp_unit_col)
        spillover_treatment_vars.append(sp_treatment_col)

        data.loc[
            data[unit_col].astype(str).isin(spillover_units),
            sp_unit_col
            ] = 1
        
        data[sp_treatment_col] = data[sp_unit_col]*data[TT_col]

    spillover_treatment_vars_join = ' + '.join(spillover_treatment_vars)

    if verbose:
        print("OK")

    return [
        data,
        spillover_treatment_vars_join,
        spillover_treatment_vars
        ]

def data_diagnostics(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    cols_relevant: list = [],
    drop_missing: bool = True,
    missing_replace_by_zero: bool = False,
    verbose: bool = config.VERBOSE
    ):

    modeldata_ismissing = tools.is_missing(
        data, 
        drop_missing = drop_missing,
        missing_replace_by_zero = missing_replace_by_zero,
        verbose = verbose
        )    

    if modeldata_ismissing[0]:        

        if drop_missing or missing_replace_by_zero:
            data = modeldata_ismissing[2]           

        if not drop_missing and not missing_replace_by_zero:
            print("WARNING: Missing values are not cleaned. Model may crash.")   
    
    other_cols_relevant = [col for col in cols_relevant if col not in [unit_col, time_col, outcome_col]]

    modeldata_isbalanced = tools.is_balanced(
        data = data,
        unit_col = unit_col,
        time_col = time_col,
        outcome_col = outcome_col,
        other_cols = other_cols_relevant,
        verbose = verbose
        )
    
    modeldata_isprepost = tools.is_prepost(
        data = data,
        unit_col = unit_col,
        time_col = time_col
        )
    if modeldata_isprepost:
        data_type = config.PREPOST_PANELDATA_DESCRIPTION
    else:
        data_type = config.MULTIPERIOD_PANELDATA_DESCRIPTION
    
    observations = len(data)
    
    outcome_mean = np.mean(data[outcome_col])
    outcome_sd = np.std(data[outcome_col])
    
    data_diagnostics_results = {
        list(config.DATA_DIAGNOSTICS)[0]: bool(modeldata_isbalanced),
        list(config.DATA_DIAGNOSTICS)[1]: bool(modeldata_ismissing[0]),
        list(config.DATA_DIAGNOSTICS)[2]: bool(drop_missing),
        list(config.DATA_DIAGNOSTICS)[3]: bool(missing_replace_by_zero),
        list(config.DATA_DIAGNOSTICS)[4]: bool(modeldata_isprepost),
        list(config.DATA_DIAGNOSTICS)[5]: data_type,
        list(config.DATA_DIAGNOSTICS)[6]: outcome_col, 
        list(config.DATA_DIAGNOSTICS)[7]: f"Mean={round(outcome_mean, config.ROUND_STATISTIC)} SD={round(outcome_sd, config.ROUND_STATISTIC)}",
        list(config.DATA_DIAGNOSTICS)[8]: observations,        
        }
    
    return data_diagnostics_results

def treatment_diagnostics(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: list,
    outcome_col: str,
    pre_post: bool = False,
    confint_alpha = 0.05,
    verbose: bool = config.VERBOSE    
    ):
    
    if verbose:
        print("Treatment diagnostics:")

    no_treatments = len(treatment_col)

    treatment_diagnostics_results = {}

    staggered_adoption = False

    for i, treatment in enumerate(treatment_col):
            
        is_notreatment_result = tools.is_notreatment(
            data = data,
            unit_col = unit_col,
            treatment_col = treatment,
            verbose = verbose 
            )
        treatment_group_size = int(len(is_notreatment_result[1]))
        control_group_size = int(len(is_notreatment_result[2]))
              
        is_parallel_result = tools.is_parallel(
            data = data,
            unit_col = unit_col,
            time_col = time_col,
            treatment_col = treatment,
            outcome_col = outcome_col,
            pre_post = pre_post,
            alpha = confint_alpha,
            verbose = verbose
            )
        
        is_simultaneous_result = tools.is_simultaneous(
            data = data,
            unit_col = unit_col,
            time_col = time_col,
            treatment_col = treatment,
            verbose = verbose
            )
        if is_simultaneous_result:
            adoption_type = config.TREATMENT_SIMULTANEOUS_DESCRIPTION
        else:
            adoption_type = config.TREATMENT_STAGGERED_DESCRIPTION
                
        is_binary_result = tools.is_binary(
            data = data,
            treatment_col = treatment,
            verbose = verbose
            )
        
        is_multiple_treatment_period_result = tools.is_multiple_treatment_period(
            data = data,
            unit_col = unit_col,
            treatment_col = treatment,
            verbose = verbose
        )
        
        treatment_diagnostics_results[i] = { 
            list(config.TREATMENT_DIAGNOSTICS)[0]: treatment,
            list(config.TREATMENT_DIAGNOSTICS)[1]: bool(is_notreatment_result[0]),
            list(config.TREATMENT_DIAGNOSTICS)[2]: is_notreatment_result[1],
            list(config.TREATMENT_DIAGNOSTICS)[3]: is_notreatment_result[2],
            list(config.TREATMENT_DIAGNOSTICS)[4]: bool(is_parallel_result[0]),
            list(config.TREATMENT_DIAGNOSTICS)[5]: bool(is_simultaneous_result),
            list(config.TREATMENT_DIAGNOSTICS)[6]: adoption_type,
            list(config.TREATMENT_DIAGNOSTICS)[7]: bool(is_binary_result[0]),
            list(config.TREATMENT_DIAGNOSTICS)[8]: is_binary_result[1],
            list(config.TREATMENT_DIAGNOSTICS)[9]: treatment_group_size,
            list(config.TREATMENT_DIAGNOSTICS)[10]: control_group_size,
            list(config.TREATMENT_DIAGNOSTICS)[11]: bool(is_multiple_treatment_period_result[0])
            }

    staggered_count = 0
    for key, value in treatment_diagnostics_results.items():
        if not value["is_simultaneous"]:
            staggered_count = staggered_count+1
    if staggered_count > 0:
        staggered_adoption = True

    untreated = tools.untreated_units(
        data = data,
        unit_col = unit_col,
        treatment_col = treatment_col,
        verbose = verbose
        )
    
    if verbose:
        print(f"There are {no_treatments} treatments (simultaneous: {no_treatments-staggered_count}, staggered: {staggered_count}) with {untreated[0]} treated and {untreated[1]} untreated units.")

    return [
        treatment_diagnostics_results,
        staggered_adoption,
        untreated
    ]

def ols_fit(
    data,
    formula,
    confint_alpha = 0.05,
    cluster_SE_by: str = None,
    verbose: bool = config.VERBOSE
    ):
    
    if verbose:
        print("Estimating model via Ordinary Least Squares", end = " ... ")  
    
    if cluster_SE_by is not None:
    
        ols_model = ols(
            formula, 
            data=data
            ).fit(
                cov_type="cluster", 
                cov_kwds={"groups": data[cluster_SE_by]} if cluster_SE_by else None
                )

    else:
        
        ols_model = ols(formula, data).fit()
        
    ols_coef = ols_model.params
    ols_coef_se = ols_model.bse
    ols_coef_t = ols_model.tvalues
    ols_coef_p = ols_model.pvalues
    ols_coef_ci = ols_model.conf_int(alpha = confint_alpha)
    
    ols_predictions = ols_model.get_prediction(data).summary_frame(alpha=confint_alpha)
    
    if verbose:
        print("OK")
    
    return [
        ols_model,
        ols_coef,
        ols_coef_se,
        ols_coef_t,
        ols_coef_p,
        ols_coef_ci,
        ols_predictions
    ]
    
def ml_fit(
    data,
    formula,
    confint_alpha = 0.05,
    family=sm.families.Gaussian(),
    link=sm.families.links.Identity(),
    verbose: bool = config.VERBOSE
    ):
    
    if verbose:
        print("Estimating model via Maximum Likelihood", end = " ... ")  
    
    y, X = dmatrices(
        formula, 
        data=data, 
        return_type = "dataframe"
        )

    mle_model = sm.GLM(
        y, 
        X, 
        family=family(link=link)
        ).fit()

    mle_coef = mle_model.params
    mle_coef_se = mle_model.bse
    mle_coef_z = mle_model.tvalues
    mle_coef_p = mle_model.pvalues
    mle_coef_ci = mle_model.conf_int(alpha=confint_alpha)
    
    mle_predictions = mle_model.get_prediction(X).summary_frame(alpha=confint_alpha)

    if verbose:
        print("OK")
        
    return [
        mle_coef,
        mle_coef_se,
        mle_coef_z,
        mle_coef_p,
        mle_coef_ci,
        mle_predictions
    ]
    
def extract_model_results(
    fit_result,
    TG_col: list = [],
    TT_col: list = [],
    treatment_col: list = [],
    after_treatment_col: list = [],
    ATT_col: list = [],
    spillover_vars: list = [],
    FE_unit_vars: list = [],
    dummy_unit_original: list = [],
    FE_time_vars: list = [],
    dummy_time_original: list = [],
    FE_group_vars: list = [],
    dummy_group_original: list = [],
    ITE_vars: list = [],
    GTE_vars: list = [],
    ITT_vars: list = [],
    GTT_vars: list = [],
    TG_x_BG_x_TT_col: list = [],
    BG_col: list = [], 
    TG_x_BG_col: list = [], 
    BG_x_TT_col: list = [],
    covariates: list = [],
    verbose: bool = config.VERBOSE
    ):
    
    if verbose:
        print("Compiling model results", end = " ... ")

    coefficients = fit_result[1]
    coef_standarderrors = fit_result[2]
    coef_t = fit_result[3]
    coef_p = fit_result[4]
    coef_conf_intervals = fit_result[5]
    
    no_treatments = len(treatment_col)
    
    model_results = {}

    if (len(treatment_col) > 0) and (any(col in coefficients for col in treatment_col)):
        
        ATE = {}
        
        for i, treatment in enumerate(treatment_col):
            ATE[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: treatment,
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[treatment],
                "SE": float(coef_standarderrors[treatment]),
                "t": float(coef_t[treatment]),
                "p": float(coef_p[treatment]),
                "CI_lower": float(coef_conf_intervals.loc[treatment, 0]),
                "CI_upper": float(coef_conf_intervals.loc[treatment, 1]),
                }
        
        model_results = {config.EFFECTS_TYPES["ATE"]["model_results_key"]: ATE}
    
    if (len(TG_col) > 0) and (any(col in coefficients for col in TG_col)):
        
        beta_1 = {}        
        
        for i, TG_ in enumerate(TG_col):
            beta_1[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: TG_,
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[TG_],
                "SE": float(coef_standarderrors[TG_]),
                "t": float(coef_t[TG_]),
                "p": float(coef_p[TG_]),
                "CI_lower": float(coef_conf_intervals.loc[TG_, 0]),
                "CI_upper": float(coef_conf_intervals.loc[TG_, 1]),
                }
                        
        model_results[config.EFFECTS_TYPES["beta_1"]["model_results_key"]] = beta_1

    if (len(TT_col) > 0) and (any(col in coefficients for col in TT_col)):
        
        delta_0 = {}
        
        for i, TT_ in enumerate(TT_col):
            delta_0[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: TT_,
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[TT_],
                "SE": float(coef_standarderrors[TT_]),
                "t": float(coef_t[TT_]),
                "p": float(coef_p[TT_]),
                "CI_lower": float(coef_conf_intervals.loc[TT_, 0]),
                "CI_upper": float(coef_conf_intervals.loc[TT_, 1]),
                }
        
        model_results[config.EFFECTS_TYPES["delta_0"]["model_results_key"]] = delta_0

    if "Intercept" in coefficients:
        
        beta_0 = {}        
       
        beta_0[0] = {
            config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients["Intercept"], 
            "SE": coef_standarderrors["Intercept"], 
            "t": coef_t["Intercept"], 
            "p": coef_p["Intercept"],
            "CI_lower": coef_conf_intervals.loc["Intercept", 0],
            "CI_upper": coef_conf_intervals.loc["Intercept", 1],
            }
        model_results[config.EFFECTS_TYPES["beta_0"]["model_results_key"]] = beta_0 
   
    if (len(after_treatment_col) > 0) and (any(col in coefficients for col in after_treatment_col)):
        
        AATE = {}
        
        for i, AATE_ in enumerate(after_treatment_col):
            AATE[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: AATE_,
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[AATE_],
                "SE": float(coef_standarderrors[AATE_]),
                "t": float(coef_t[AATE_]),
                "p": float(coef_p[AATE_]),
                "CI_lower": float(coef_conf_intervals.loc[AATE_, 0]),
                "CI_upper": float(coef_conf_intervals.loc[AATE_, 1]),
                }
            
        model_results[config.EFFECTS_TYPES["AATE"]["model_results_key"]] = AATE
    
    if (len(ATT_col) > 0) and (any(col in coefficients for col in ATT_col)):
        
        ATT = {}
                
        for i, ATT_ in enumerate(ATT_col):
            ATT[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: ATT_,
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[ATT_],
                "SE": float(coef_standarderrors[ATT_]),
                "t": float(coef_t[ATT_]),
                "p": float(coef_p[ATT_]),
                "CI_lower": float(coef_conf_intervals.loc[ATT_, 0]),
                "CI_upper": float(coef_conf_intervals.loc[ATT_, 1]),
                }
                       
        model_results[config.EFFECTS_TYPES["ATT"]["model_results_key"]] = ATT    
    
    if (len(spillover_vars) > 0) and (any(col in coefficients for col in spillover_vars)):

        spillover_coef = {}
        
        for i, spillover_var in enumerate(spillover_vars):
            spillover_coef[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: spillover_var,
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[spillover_var],
                "SE": float(coef_standarderrors[spillover_var]),
                "t": float(coef_t[spillover_var]),
                "p": float(coef_p[spillover_var]),
                "CI_lower": float(coef_conf_intervals.loc[spillover_var, 0]),
                "CI_upper": float(coef_conf_intervals.loc[spillover_var, 1]),
                } 
                       
        model_results[config.EFFECTS_TYPES["spillover"]["model_results_key"]] = spillover_coef
    
    fixed_effects = [
        None, 
        None, 
        None
        ]  

    if (len(FE_unit_vars) > 0) and (any(col in coefficients for col in FE_unit_vars)):
        
        FE_unit_coef = {}
               
        for i, unit_dummy in enumerate(FE_unit_vars):
            FE_unit_coef[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: dummy_unit_original[i],
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[unit_dummy],
                "SE": float(coef_standarderrors[unit_dummy]),
                "t": float(coef_t[unit_dummy]),
                "p": float(coef_p[unit_dummy]),
                "CI_lower": float(coef_conf_intervals.loc[unit_dummy, 0]),
                "CI_upper": float(coef_conf_intervals.loc[unit_dummy, 1]),
                "Coefficient_type": config.EFFECTS_TYPES["FE"]["types"][0]["description"]
                }
            
        fixed_effects[0] = {config.EFFECTS_TYPES["FE"]["types"][0]["model_results_key"]: FE_unit_coef}
    
    if (len(FE_time_vars) > 0) and (any(col in coefficients for col in FE_time_vars)):

        FE_time_coef = {}
        
        for i, time_dummy in enumerate(FE_time_vars):
            FE_time_coef[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: dummy_time_original[i],
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[time_dummy],
                "SE": float(coef_standarderrors[time_dummy]),
                "t": float(coef_t[time_dummy]),
                "p": float(coef_p[time_dummy]),
                "CI_lower": float(coef_conf_intervals.loc[time_dummy, 0]),
                "CI_upper": float(coef_conf_intervals.loc[time_dummy, 1]),
                "Coefficient_type": config.EFFECTS_TYPES["FE"]["types"][1]["description"]
                }
            
        fixed_effects[1] = {config.EFFECTS_TYPES["FE"]["types"][1]["model_results_key"]: FE_time_coef}
        
    if (len(FE_group_vars) > 0) and (any(col in coefficients for col in FE_group_vars)):

        FE_group_coef = {}
        
        for i, group_dummy in enumerate(FE_group_vars):
            FE_group_coef[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: dummy_group_original[i],
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[group_dummy],
                "SE": float(coef_standarderrors[group_dummy]),
                "t": float(coef_t[group_dummy]),
                "p": float(coef_p[group_dummy]),
                "CI_lower": float(coef_conf_intervals.loc[group_dummy, 0]),
                "CI_upper": float(coef_conf_intervals.loc[group_dummy, 1]),
                "Coefficient_type": config.EFFECTS_TYPES["FE"]["types"][2]["description"]
                }
            
        fixed_effects[2] = {config.EFFECTS_TYPES["FE"]["types"][2]["model_results_key"]: FE_group_coef}

    model_results[config.EFFECTS_TYPES["FE"]["model_results_key"]] = fixed_effects

    if (len(ITT_vars) > 0) and (any(col in coefficients for col in ITT_vars)):
      
        ITT_coef = {}

        for i, ITT_var in enumerate(ITT_vars):

            ITT_coef[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: dummy_unit_original[i],
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[ITT_var],
                "SE": float(coef_standarderrors[ITT_var]),
                "t": float(coef_t[ITT_var]),
                "p": float(coef_p[ITT_var]),
                "CI_lower": float(coef_conf_intervals.loc[ITT_var, 0]),
                "CI_upper": float(coef_conf_intervals.loc[ITT_var, 1]),
                }      
        
        model_results["individual_time_trends"] = ITT_coef

    if (len(ITE_vars) > 0) and (any(col in coefficients for col in ITE_vars)):
    
        ITE_coef = {}

        dummy_unit_current = dummy_unit_original*no_treatments

        treatment_current = []
        for treatment in treatment_col:
            treatment_single = [treatment]*len(dummy_unit_original)
            treatment_current = treatment_current + treatment_single

        for i, ITE_var in enumerate(ITE_vars):                

            ITE_coef[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: treatment_current[i] + " " + dummy_unit_current[i],
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: float(coefficients[ITE_var]),
                "SE": float(coef_standarderrors[ITE_var]),
                "t": float(coef_t[ITE_var]),
                "p": float(coef_p[ITE_var]),
                "CI_lower": float(coef_conf_intervals.loc[ITE_var, 0]),
                "CI_upper": float(coef_conf_intervals.loc[ITE_var, 1]),
                }            
            
        model_results["individual_treatment_effects"] = ITE_coef

    if (len(GTT_vars) > 0) and (any(col in coefficients for col in GTT_vars)):
        
        GTT_coef = {}

        for i, GTT_var in enumerate(GTT_vars):

            GTT_coef[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: dummy_group_original[i],
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[GTT_var],
                "SE": float(coef_standarderrors[GTT_var]),
                "t": float(coef_t[GTT_var]),
                "p": float(coef_p[GTT_var]),
                "CI_lower": float(coef_conf_intervals.loc[GTT_var, 0]),
                "CI_upper": float(coef_conf_intervals.loc[GTT_var, 1]),
                }      
        
        model_results["group_time_trends"] = GTT_coef

    if (len(GTE_vars) > 0) and (any(col in coefficients for col in GTE_vars)):
        
        GTE_coef = {}

        dummy_group_current = dummy_group_original*no_treatments

        treatment_current = []
        for treatment in treatment_col:
            treatment_single = [treatment]*len(dummy_group_original)
            treatment_current = treatment_current + treatment_single
        
        for i, GTE_var in enumerate(GTE_vars):
            GTE_coef[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: treatment_current[i] + " " + dummy_group_current[i],
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[GTE_var],
                "SE": float(coef_standarderrors[GTE_var]),
                "t": float(coef_t[GTE_var]),
                "p": float(coef_p[GTE_var]),
                "CI_lower": float(coef_conf_intervals.loc[GTE_var, 0]),
                "CI_upper": float(coef_conf_intervals.loc[GTE_var, 1]),
                }      
        
        model_results["group_treatment_effects"] = GTE_coef

    if (len(covariates) > 0) and (any(col in coefficients for col in covariates)):

        covariates_effects = {}
        
        for i, covariate in enumerate(covariates):
            covariates_effects[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: covariate,
                config.OLS_MODEL_RESULTS["coef"]["model_results_key"]: coefficients[covariate],
                "SE": float(coef_standarderrors[covariate]),
                "t": float(coef_t[covariate]),
                "p": float(coef_p[covariate]),
                "CI_lower": float(coef_conf_intervals.loc[covariate, 0]),
                "CI_upper": float(coef_conf_intervals.loc[covariate, 1]),
                }
            
        model_results["covariates_effects"] = covariates_effects    

    if (len(TG_x_BG_x_TT_col) > 0) and (any(col in coefficients for col in TG_x_BG_x_TT_col)):

        TDATE = {}

        for i, TG_x_BG_x_TT_ in enumerate(TG_x_BG_x_TT_col):
            TDATE[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: TG_x_BG_x_TT_,
                "Estimate": coefficients[TG_x_BG_x_TT_],
                "SE": float(coef_standarderrors[TG_x_BG_x_TT_]),
                "t": float(coef_t[TG_x_BG_x_TT_]),
                "p": float(coef_p[TG_x_BG_x_TT_]),
                "CI_lower": float(coef_conf_intervals.loc[TG_x_BG_x_TT_, 0]),
                "CI_upper": float(coef_conf_intervals.loc[TG_x_BG_x_TT_, 1]),
                }
        
        model_results = {config.EFFECTS_TYPES_DDD["TDATE"]["model_results_key"]: TDATE}
    
    if (len(BG_col) > 0) and (any(col in coefficients for col in BG_col)):

        BG = {}        

        for i, BG_ in enumerate(BG_col):
            BG[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: BG_,
                "Estimate": coefficients[BG_],
                "SE": float(coef_standarderrors[BG_]),
                "t": float(coef_t[BG_]),
                "p": float(coef_p[BG_]),
                "CI_lower": float(coef_conf_intervals.loc[BG_, 0]),
                "CI_upper": float(coef_conf_intervals.loc[BG_, 1]),
                }            

        model_results[config.EFFECTS_TYPES_DDD["beta_2"]["model_results_key"]] = BG
        
    if (len(TG_x_BG_col) > 0) and (any(col in coefficients for col in TG_x_BG_col)):

        TG_x_BG = {}

        for i, TG_x_BG_ in enumerate(TG_x_BG_col):
            TG_x_BG[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: TG_x_BG_,
                "Estimate": coefficients[TG_x_BG_],
                "SE": float(coef_standarderrors[TG_x_BG_]),
                "t": float(coef_t[TG_x_BG_]),
                "p": float(coef_p[TG_x_BG_]),
                "CI_lower": float(coef_conf_intervals.loc[TG_x_BG_, 0]),
                "CI_upper": float(coef_conf_intervals.loc[TG_x_BG_, 1]),
                }            

        model_results[config.EFFECTS_TYPES_DDD["beta_4"]["model_results_key"]] = TG_x_BG

    if (len(BG_x_TT_col) > 0) and (any(col in coefficients for col in BG_x_TT_col)):

        BG_x_TT = {}        

        for i, BG_x_TT_ in enumerate(BG_x_TT_col):
            BG_x_TT[i] = {
                config.OLS_MODEL_RESULTS["coef_name"]["model_results_key"]: BG_x_TT_,
                "Estimate": coefficients[BG_x_TT_],
                "SE": float(coef_standarderrors[BG_x_TT_]),
                "t": float(coef_t[BG_x_TT_]),
                "p": float(coef_p[BG_x_TT_]),
                "CI_lower": float(coef_conf_intervals.loc[BG_x_TT_, 0]),
                "CI_upper": float(coef_conf_intervals.loc[BG_x_TT_, 1]),
                }            

        model_results[config.EFFECTS_TYPES_DDD["beta_6"]["model_results_key"]] = BG_x_TT
    
    if verbose:
        print("OK")

    return model_results

def fit_metrics(
    data,
    outcome_col,
    model_predictions,
    indep_vars_no: int = None
    ):
    
    model_predictions = pd.DataFrame(model_predictions)
    model_predictions = model_predictions.reset_index()
    model_predictions.rename(columns = {config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[0]: f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}"}, inplace = True)

    observed_expected = pd.concat (
        [
            data, 
            model_predictions
            ], 
            axis = 1
            )

    fit_metrics_result = tools.fit_metrics(
        observed = observed_expected[outcome_col],
        expected = observed_expected[f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}"],
        indep_vars_no = indep_vars_no
    )

    return fit_metrics_result

def create_timestamp(function):

    now = datetime.now()

    timestamp_dict = {
        "package_version": f"{config.PACKAGE_NAME} {config.PACKAGE_VERSION}",
        "function": function,
        "datetime": now.strftime("%Y-%m-%d %H-%M-%S")
    }

    return timestamp_dict