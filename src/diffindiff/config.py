#-----------------------------------------------------------------------
# Name:        config (diffindiff package)
# Purpose:     Configuration for the diffindiff package
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.4
# Last update: 2025-12-06 11:52
# Copyright (c) 2025 Thomas Wieland
#-----------------------------------------------------------------------

# Basic config:

PACKAGE_VERSION = "2.2.2"

VERBOSE = False

ROUND_STATISTIC = 3
ROUND_PERCENT = 2

AUTO_SWITCH_TO_PREPOST = True

# Description texts:

DID_DESCRIPTION = "Difference-in-Differences Analysis"
DDD_DESCRIPTION = "Triple-Difference Analysis"

TREATMENT_DESCRIPTION = "Treatment"

GROUP_DESCRIPTION = "Group"

TREATMENT_GROUP_DESCRIPTION = f"{TREATMENT_DESCRIPTION} {GROUP_DESCRIPTION}"
CONTROL_GROUP_DESCRIPTION = f"Control {GROUP_DESCRIPTION}"
GROUPS_DESCRIPTION = f"{TREATMENT_DESCRIPTION} and {CONTROL_GROUP_DESCRIPTION}"

TIME_PERIODS_DESCRIPTION = "Time periods"
TREATMENT_PERIOD_DESCRIPTION = f"{TREATMENT_DESCRIPTION} period"
STUDY_PERIOD_DESCRIPTION = "Study period"
PREPOST_DESCRIPTION = "Pre-post"
AFTER_TREATMENT_PERIOD_DESCRIPTION = "After-treatment period"

UNITS_DESCRIPTION = "Units"

N_DESCRIPTION = "Number of observations"

DDD_GROUP_DESCRIPTION = f"{GROUP_DESCRIPTION} segmentation"

TREATMENT_SIMULTANEOUS_DESCRIPTION = "Simultaneous"
TREATMENT_STAGGERED_DESCRIPTION = "Staggered"
NO_TREATMENT_CG_DESCRIPTION = f"No-treatment {CONTROL_GROUP_DESCRIPTION}"

PREPOST_PANELDATA_DESCRIPTION = f"{PREPOST_DESCRIPTION} data"
MULTIPERIOD_PANELDATA_DESCRIPTION = "Multi-period panel data"

DIDDATA_SUMMARY_LABELS = [
    TREATMENT_DESCRIPTION,
    UNITS_DESCRIPTION,
    TREATMENT_GROUP_DESCRIPTION,
    CONTROL_GROUP_DESCRIPTION,
    DDD_GROUP_DESCRIPTION,
    STUDY_PERIOD_DESCRIPTION,
    TREATMENT_PERIOD_DESCRIPTION,
    N_DESCRIPTION
]
DIDDATA_SUMMARY_MAX_WIDTH = max(len(label) for label in DIDDATA_SUMMARY_LABELS) + 1

# Data management:

DELIMITER = "_"
DELIMITER_INTERACT = "x"

COL_ABBREV = "col"

TG_COL = "TG"
CG_COL = "CG"
BG_COL = "BG"
TT_COL = "TT"
ATT_COL = "ATT"
TIME_COL = "t"
UNIT_COL = "unit"
UNIT_TIME_COL = f"{UNIT_COL}{DELIMITER}{TIME_COL}"
TIME_COUNTER_COL = "time_counter"
TREATMENT_COL = f"{TG_COL}{DELIMITER_INTERACT}{TT_COL}"

DUMMY_PREFIX = "DUMMY"
LOG_PREFIX = "log"
OBSERVED_SUFFIX = "observed"
EXPECTED_SUFFIX = "expected"
PREDICTED_SUFFIX = "pred"
CI_LOWER_SUFFIX = "CI_lower"
CI_UPPER_SUFFIX = "CI_upper"
PI_LOWER_SUFFIX = "PI_lower"
PI_UPPER_SUFFIX = "PI_upper"
SPILLOVER_PREFIX = "Spillover"
SPILLOVER_UNIT_PREFIX = f"{SPILLOVER_PREFIX}{DELIMITER}{UNIT_COL}"

# Modeling config:

# Coefficients/effects types:

TREATMENT_EFFECTS_DESCRIPTION = "Difference-in-Differences coefficients"

EFFECTS_TYPES = {
    "ATE": {
        "description": "Average treatment effect",
        "model_results_key": "average_treatment_effects",
        "summary_treatment_effects": True,
        "summary_description": "{description} {coef}"
    },
    "AATE": {
        "description": "Average after-treatment effect",
        "model_results_key": "average_after_treatment_effects",
        "summary_treatment_effects": True,
        "summary_description": "{description} {coef}"
    },    
    "beta_0": {
        "description": "Control group baseline",
        "model_results_key": "control_group_baseline",
        "summary_treatment_effects": True,
        "summary_description": "{description}"
    },
    "beta_1": {
        "description": f"{TREATMENT_GROUP_DESCRIPTION} deviation",
        "model_results_key": "treatment_group_deviation",
        "summary_treatment_effects": True,
        "summary_description": "{description}"
    },
    "delta_0": {
        "description": "Non-treatment time effect",
        "model_results_key": "non_treatment_time_effect",
        "summary_treatment_effects": True,
        "summary_description": "{description} {coef}"
    },
    "ATT": {
        "description": "After-treatment time effect",
        "model_results_key": "after_treatment_time_effects",
        "summary_treatment_effects": True,
        "summary_description": "{description} {coef}"
    },
    "FE": {
        "description": "Fixed effects",
        "model_results_key": "fixed_effects",
        "summary_treatment_effects": False,
        "types": {
            0: {
                "FE": "unit",
                "dummy_prefix": "UNIT",
                "model_config_key": "FE_unit",
                "model_results_key": "FE_unit",
                "description": "Fixed effects for observational units"
            },
            1: {
                "FE": "time",
                "dummy_prefix": "TIME",
                "model_config_key": "FE_time",
                "model_results_key": "FE_time",
                "description": "Fixed effects for time points"
            },
            2: {
                "FE": "group",
                "dummy_prefix": "GROUP",
                "model_config_key": "FE_group",
                "model_results_key": "FE_group",
                "description": "Fixed effects for groups"
            },
        }
    },
    "ITT": {
        "description": "Individual time trends",
        "model_results_key": "individual_time_trends",
        "model_config_key": "ITT",
        "summary_treatment_effects": False
    },
    "ITE": {
        "description": "Individual treatment effects",
        "model_results_key": "individual_treatment_effects",
        "model_config_key": "ITE",
        "summary_treatment_effects": True,
        "summary_description": "{coef}"
    },
    "GTT": {
        "description": "Group time trends",
        "model_results_key": "group_time_trends",
        "model_config_key": "GTT",
        "summary_treatment_effects": False
    },
    "GTE": {
        "description": "Group treatment effects",
        "model_results_key": "group_treatment_effects",
        "model_config_key": "GTE",
        "summary_treatment_effects": True,
        "summary_description": "{coef}"
    },
    "spillover": {
        "description": "Treatment spillover effect",
        "model_results_key": "treatment_spillover_effects",
        "model_config_key": "spillover_effects",
        "summary_treatment_effects": True,
        "summary_description": "{description} {coef}"
    },
    "covariates": {
        "description": "Covariates",
        "model_results_key": "covariates_effects",
        "model_config_key": "covariates",
        "summary_treatment_effects": False,
        "summary_description": "{coef}"
    }, 
}
EFFECTS_TYPES_MODEL_RESULTS = [value["model_results_key"] for value in EFFECTS_TYPES.values() if "model_results_key" in value]
EFFECTS_TYPES_MODEL_RESULTS_SUMMARY = [value["model_results_key"] for value in EFFECTS_TYPES.values() if "model_results_key" in value and value["summary_treatment_effects"]]


EFFECTS_TYPES_DDD = {
    "TDATE": {
        "description": "Triple-Difference Average treatment effect",
        "model_results_key": "TDATE"
    },
    "beta_2": {
        "description": "Benefit group deviation",
        "model_results_key": "benefit_group_deviation"
    },
    "beta_4": {
        "description": "Treated benefit group deviation",
        "model_results_key": "treated_benefit_group_deviation"
    },
    "beta_6": {
        "description": "Benefit group non-treatment time effect",
        "model_results_key": "benefit_non_treatment_time_effect"
    },
    list(EFFECTS_TYPES.keys())[0]: EFFECTS_TYPES[list(EFFECTS_TYPES.keys())[0]],
    list(EFFECTS_TYPES.keys())[2]: EFFECTS_TYPES[list(EFFECTS_TYPES.keys())[2]],
    list(EFFECTS_TYPES.keys())[3]: EFFECTS_TYPES[list(EFFECTS_TYPES.keys())[3]],
    list(EFFECTS_TYPES.keys())[4]: EFFECTS_TYPES[list(EFFECTS_TYPES.keys())[4]],
    list(EFFECTS_TYPES.keys())[6]: EFFECTS_TYPES[list(EFFECTS_TYPES.keys())[6]],
}

EFFECTS_TYPES_DDD_MODEL_RESULTS = [value["model_results_key"] for value in EFFECTS_TYPES_DDD.values() if "model_results_key" in value]

FE_TYPES = [value["FE"] for value in EFFECTS_TYPES["FE"]["types"].values()]

# Time trends:
TIME_TRENDS_TYPES = [
    list(EFFECTS_TYPES.keys())[7],
    list(EFFECTS_TYPES.keys())[9]
    ]

# Specific effects:
SPECIFIC_EFFFECTS_TYPES = [
    list(EFFECTS_TYPES.keys())[8],
    list(EFFECTS_TYPES.keys())[10]
    ]

OLS_MODEL_RESULTS = {
    "coef_name": {
        "model_results_key": "Coefficient",
        "summary_description": "Coefficient",
    },
    "coef": {
        "model_results_key": "Estimate",
        "summary_description": "Estimate",
    },
    "coef_standard_errors": {
        "model_results_key": "SE",
        "summary_description": "SE",
    },
    "coef_teststatistic": {
        "model_results_key": "t",
        "summary_description": "t"
    },
    "coef_p": {
        "model_results_key": "p",
        "summary_description": "p"
    },
    "coef_confint_lower": {
        "model_results_key": "CI_lower",
        "summary_description": "CI lower"
    },
    "coef_confint_upper": {
        "model_results_key": "CI_upper",
        "summary_description": "CI upper"
    },    
}

ML_MODEL_RESULTS = {
    "coef_name": {
        "model_results_key": OLS_MODEL_RESULTS["coef_name"]["model_results_key"],
        "summary_description": OLS_MODEL_RESULTS["coef_name"]["summary_description"],
    },
    "coef": {
        "model_results_key": OLS_MODEL_RESULTS["coef"]["model_results_key"],
        "summary_description": OLS_MODEL_RESULTS["coef"]["summary_description"],
    },
    "coef_standard_errors": {
        "model_results_key": OLS_MODEL_RESULTS["coef_standard_errors"]["model_results_key"],
        "summary_description": OLS_MODEL_RESULTS["coef_standard_errors"]["summary_description"],
    },
    "coef_teststatistic": {
        "model_results_key": "z",
        "summary_description": "z"
    },
    "coef_p": {
        "model_results_key": OLS_MODEL_RESULTS["coef_p"]["model_results_key"],
        "summary_description": OLS_MODEL_RESULTS["coef_p"]["summary_description"]
    },
    "coef_confint_lower": {
        "model_results_key": OLS_MODEL_RESULTS["coef_confint_lower"]["model_results_key"],
        "summary_description": OLS_MODEL_RESULTS["coef_confint_lower"]["summary_description"]
    },
    "coef_confint_upper": {
        "model_results_key": OLS_MODEL_RESULTS["coef_confint_upper"]["model_results_key"],
        "summary_description": OLS_MODEL_RESULTS["coef_confint_upper"]["summary_description"]
    },    
}

# Model fit metrics:

MODEL_FIT_METRICS_DESCRIPTION = "Model fit metric"

MODEL_FIT_METRICS = {
    "SSR": {
        "description": "Sum of squared residuals",
        "show_in_summary": False
        },
    "SAR": {
        "description": "Sum of absolute residuals",
        "show_in_summary": False
        },
    "SQT": {
        "description": "Total variance of dependent variable",
        "show_in_summary": False
        },
    "RSQ": {
        "description": "R-Squared",
        "show_in_summary": True
        },
    "RSQ_ADJ": {
        "description": "R-Squared adjusted",
        "show_in_summary": True
        },
    "MSE": {
        "description": "Mean squared error",
        "show_in_summary": True
        },
    "RMSE": {
        "description": "Root mean squared error",
        "show_in_summary": True
        },
    "MAE": {
        "description": "Mean absolute error",
        "show_in_summary": True
        },
    "MAPE": {
        "description": "Mean absolute percentage error",
        "show_in_summary": True
        },
    "MAPE_SYM": {
        "description": "Symmetric mean absolute percentage error",
        "show_in_summary": False
        }
}

# Treatment diagnostics:

TREATMENT_DIAGNOSTICS_DESCRIPTION = f"{TREATMENT_DESCRIPTION} diagnostics"

TREATMENT_DIAGNOSTICS = {
    "treatment": {
        "description": TREATMENT_DESCRIPTION,
        "show_in_summary": True
        },
    "is_notreatment": {
        "description": NO_TREATMENT_CG_DESCRIPTION,
        "show_in_summary": True,
        },
    "treatment_group": {
        "description": TREATMENT_GROUP_DESCRIPTION,
        "show_in_summary": False
        },
    "control_group": {
        "description": CONTROL_GROUP_DESCRIPTION,
        "show_in_summary": False
        },
    "is_parallel": {
        "description": "Parallel trends (pre)",
        "show_in_summary": True
        },
    "is_simultaneous": {
        "description": f"{TREATMENT_SIMULTANEOUS_DESCRIPTION} treatment",
        "show_in_summary": False
    },
    "adoption_type": {
        "description": "Type of adoption",
        "show_in_summary": True
    },
    "is_binary": {
        "description": "Binary treatment",
        "show_in_summary": False
    },
    "treatment_format": {
        "description": f"{TREATMENT_DESCRIPTION} format",
        "show_in_summary": True
    },
    "treatment_group_size": {
        "description": f"{TREATMENT_GROUP_DESCRIPTION} (N)",
        "show_in_summary": True
    },
    "control_group_size": {
        "description": f"{CONTROL_GROUP_DESCRIPTION} (N)",
        "show_in_summary": True
    },
    "is_multiple_treatment_period": {
        "description": "Multiple treatment periods",
        "show_in_summary": True
    },
}

# Input data diagnostics:

DATA_DIAGNOSTICS_DESCRIPTION = "Input data diagnostics"

DATA_DIAGNOSTICS = {
    "is_balanced": {
        "description": "Balanced panel data",
        "show_in_summary": True
        },
    "is_missing": {
        "description": "Missing values",
        "show_in_summary": True
        },
    "drop_missing": {
        "description": "Drop missing values",
        "show_in_summary": False
        }, 
    "missing_replace_by_zero": {
        "description": "Replace missing values by zero",
        "show_in_summary": False
        },
    "is_prepost": {
        "description": PREPOST_PANELDATA_DESCRIPTION,
        "show_in_summary": False
        },
    "data_type": {
        "description": "Data type",
        "show_in_summary": True
        },
    "outcome_col": {
        "description": "Outcome variable",
        "show_in_summary": True
        },
    "outcome_descriptives": {
        "description": "Outcome descriptives",
        "show_in_summary": True
        },
    "observations": {
        "description": N_DESCRIPTION,
        "show_in_summary": True
        },
    }

DIAGNOSTICS_COLUMN = "Result"

COVARIATES_DESCRIPTION = "Covariates"

# Predictions:

PREDICTIONS_SUMMARY_FRAME_COLS = {
    "mean": "Predicted mean",
    "mean_se": "Predicted mean SE",
    "mean_ci_lower": "Lower CI of mean",
    "mean_ci_upper": "Upper CI of mean",
    "obs_ci_lower": "Lower prediction interval",
    "obs_ci_upper": "Upper prediction interval",
}
PREDICTIONS_SUMMARY_FRAME_COLS_LIST = list(PREDICTIONS_SUMMARY_FRAME_COLS.keys())
PREDICTIONS_SUMMARY_FRAME_DESCRIPTIONS = list(PREDICTIONS_SUMMARY_FRAME_COLS.values())

# Counterfactual:
COUNTERFAC_SUFFIX_CF = "counterfac"
COUNTERFAC_SUFFIX_PRED_CF = f"{DELIMITER}{PREDICTED_SUFFIX}{DELIMITER}{COUNTERFAC_SUFFIX_CF}"