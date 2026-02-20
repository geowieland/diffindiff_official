#-----------------------------------------------------------------------
# Name:        diddata (diffindiff package)
# Purpose:     Creating data for Difference-in-Differences Analysis
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     2.1.6
# Last update: 2026-02-20 18:28
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import diffindiff.didanalysis as didanalysis
import diffindiff.didtools as tools
import diffindiff.config as config
import diffindiff.didanalysis_helper as helper

class DiffGroups:
    def __init__(
        self, 
        groups_data: list, 
        groups_config_dict: dict,
        timestamp: dict
        ):

        self.data = [
            groups_data, 
            groups_config_dict,
            timestamp
            ]

    def get_data(self):
        return self.data[0]

    def get_config(self):
        return self.data[1]
    
    def get_timestamp(self):
        return self.data[2]

    def summary(self):

        groups_config = self.data[1]

        title = f"{config.DID_DESCRIPTION} {config.GROUPS_DESCRIPTION}"
        total_width = len(title)

        print("=" * total_width)
        print(title)
        print("-" * total_width)

        for key, value in groups_config.items():
            
            if value["treatment_name"] is not None:
                print (f"{config.TREATMENT_DESCRIPTION} {value['treatment_name']}")
            else:
                print (f"{config.TREATMENT_DESCRIPTION} {key+1}")
            
            print (f" {config.UNITS_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {value['full_sample']} ({round(value['full_sample']/value['full_sample']*100, config.ROUND_PERCENT)} %)")
            print (f" {config.TREATMENT_GROUP_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {value['treatment_group']} ({round(value['treatment_group']/value['full_sample']*100, config.ROUND_PERCENT)} %)")
            print (f" {config.CONTROL_GROUP_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {value['control_group']} ({round(value['control_group']/value['full_sample']*100, config.ROUND_PERCENT)} %)")
            
            if groups_config[key]["DDD"]:
                print (f" {config.DDD_GROUP_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} YES")
            else:
                print (f" {config.DDD_GROUP_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} NO")
        
        print("=" * total_width)

    def add_segmentation(
        self,
        group_benefit: list,
        verbose: bool = config.VERBOSE       
        ):

        groups_config = self.data[1]        
        
        if groups_config["DDD"]:
            
            print("DiffGroups object already includes a benefit group. No segmentation added.")
            
            groups = self
            
        else:

            if verbose:
                print(f"Adding benefit group with {len(group_benefit)} units to groups data", end = " ... ")                
            
            groups_data = self.data[0]
            
            groups_data[config.BG_COL] = 0
            groups_data.loc[groups_data[config.UNIT_COL].astype(str).isin(group_benefit), config.BG_COL] = 1
            
            groups_config["DDD"] = True

            groups = DiffGroups(
                groups_data, 
                groups_config,
                timestamp = helper.create_timestamp(function="add_segmentation")
                )

            if verbose:
                print("OK")

        return groups

def create_groups(
    treatment_group,
    control_group,
    treatment_name: str = None,
    verbose: bool = config.VERBOSE
    ):

    TG_col = config.TG_COL
    
    if treatment_name is not None:
        
        if not isinstance(treatment_name, str):
            raise TypeError(f"Parameter 'treatment_name' must be stated as str")
        
        treatment_name = tools.clean_treatment_name(treatment_name)
        
        if verbose:
            print(f"Creating groups for treatment '{treatment_name}'", end = " ... ")
            
        TG_col = f"{config.TG_COL}{config.DELIMITER}{treatment_name}"
    
    else:
        
        if verbose:
            print("Creating groups for unnamed treatment", end = " ... ")
    
    treatment_group_unique = tools.unique(treatment_group)
    control_group_unique = tools.unique(control_group)
    
    treatment_group_N = len(treatment_group_unique)
    control_group_N = len(control_group_unique)

    TG_dummies = [1] * treatment_group_N
    CG_dummies = [0] * control_group_N

    TG_data = {
        config.UNIT_COL: treatment_group_unique, 
        TG_col: TG_dummies
        }
    CG_data = {
        config.UNIT_COL: control_group_unique, 
        TG_col: CG_dummies
        }

    groups_data = []
    groups_data = [
        pd.concat(
            [
                pd.DataFrame(TG_data),
                pd.DataFrame(CG_data)
            ], 
        axis = 0
        )
        ]
    
    DDD = False
    own_counterfactual = False

    groups_config = {
        0: {
            "treatment_group": treatment_group_N,
            "control_group": control_group_N,
            "full_sample": treatment_group_N+control_group_N,
            "DDD": DDD,
            "own_counterfactual": own_counterfactual,
            "TG_col": TG_col,
            "BG_col": None,
            "treatment_name": treatment_name,            
            }
            }

    groups = DiffGroups(
        groups_data, 
        groups_config,
        timestamp = helper.create_timestamp(function="create_groups")
        )
    
    if verbose:
        print("OK")

    return groups

class DiffTreatment:

    def __init__(
        self, 
        treatment_data_df: pd.DataFrame, 
        treatment_config_dict: dict, 
        treatment_meta: dict,
        timestamp: dict
        ):

        self.data = [
            treatment_data_df, 
            treatment_config_dict, 
            treatment_meta,
            timestamp
            ]

    def get_data(self):
        return self.data[0]

    def get_config(self):
        return self.data[1]
    
    def get_metadata(self):
        return self.data[2]
    
    def get_timestamp(self):
        return self.data[3]

    def summary(self):

        treatment_config = self.data[1]
        treatment_meta = self.data[2]

        title = f"{config.DID_DESCRIPTION} {config.TREATMENT_DESCRIPTION} Configuration"
        total_width = len(title)

        print("=" * total_width)
        print(title)
        print("-" * total_width)
        
        print(f"{config.STUDY_PERIOD_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}}  {treatment_meta['study_period_start']} - {treatment_meta['study_period_end']}")
        print("")

        for key, value in treatment_config.items():
            
            if value["treatment_name"] is not None:
                print(f"{config.TREATMENT_DESCRIPTION} {value['treatment_name']}")
            else:
                print(f"{config.TREATMENT_DESCRIPTION} {key+1}")
            
            if treatment_meta["pre_post"]:
                print(f" {config.PREPOST_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {value['treatment_period_start']} vs. {value['treatment_period_end']}")
            else:
                print(f" {config.TREATMENT_PERIOD_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {value['treatment_period_start']} - {value['treatment_period_end']} ({value['treatment_period']} {treatment_meta['frequency']})")
            
            if treatment_config[key]["after_treatment_period"]:
                print(f" {config.AFTER_TREATMENT_PERIOD_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {value['treatment_period_end']} - {value['study_period_end']} ({value['after_treatment_period_N']} {treatment_meta['frequency']})")

        print("=" * total_width)

def create_treatment(
    study_period,
    treatment_period,
    freq = "D",
    date_format = "%Y-%m-%d",
    treatment_name: str = None,
    pre_post: bool = False,
    after_treatment_period: bool = False,
    verbose = config.VERBOSE
    ): 
       
    check_dates = tools.check_date_format(
        dates = study_period+treatment_period,
        date_format = date_format
    )
    if check_dates[0]:
        raise ValueError(f"Study and/or treatment period include invalid dates: {', '.join(check_dates[1])}.")

    TT_col = config.TT_COL
    
    if treatment_name is not None:
        
        if not isinstance(treatment_name, str):
            raise TypeError(f"Parameter 'treatment_name' must be stated as str")
        
        treatment_name = tools.clean_treatment_name(treatment_name)
        
        if verbose:
            print(f"Creating treatment data for treatment '{treatment_name}'", end = " ... ")
    
        TT_col = f"{config.TT_COL}{config.DELIMITER}{treatment_name}"
            
    else:
        
        if verbose:
            print("Creating treatment data for unnamed treatment", end = " ... ")
            
    if after_treatment_period:
        
        if treatment_name is not None:
            ATT_col = f"{config.ATT_COL}{config.DELIMITER}{treatment_name}"
            
        else:
            ATT_col = config.ATT_COL
    else:
        ATT_col = None

    if pre_post:

        after_treatment_period = False

        study_period_range = [study_period[0], study_period[1]]
        study_period_N = 2
        study_period_counter = [1,2]

        treatment_period_range = [treatment_period[0], treatment_period[1]]
        treatment_period_N = 1
        TT_dummies = [0,1]

        study_period_range = pd.DataFrame (treatment_period_range, columns=[config.TIME_COL])
        study_period_range[config.TIME_COUNTER_COL] = pd.DataFrame(study_period_counter)

        TT_data = {
            config.TIME_COL: treatment_period_range, 
            TT_col: TT_dummies
            }
        
        TT_data = pd.DataFrame(TT_data)
        
        treatment_period_range = pd.DataFrame(
            study_period_range
            )

        treatment_data = treatment_period_range.merge(TT_data, how = "left")

    else:

        study_period_range = pd.date_range(
            start = study_period[0], 
            end = study_period[1], 
            freq = freq
            )

        study_period_N = len(study_period_range)
        study_period_counter = np.arange (1, study_period_N+1, 1)

        treatment_period_range = pd.date_range(
            start = treatment_period[0], 
            end = treatment_period[1],
            freq = freq
            )

        treatment_period_N = len(treatment_period_range)

        TT_dummies = [1] * treatment_period_N

        study_period_range = {config.TIME_COL: study_period_range}
        study_period_range = pd.DataFrame (study_period_range)
        study_period_range[config.TIME_COUNTER_COL] = pd.DataFrame(study_period_counter)

        TT_data = {
            config.TIME_COL: treatment_period_range, 
            TT_col: TT_dummies
            }
        TT_data = pd.DataFrame(TT_data)
    
        treatment_data = study_period_range.merge(
            TT_data, 
            how = "left"
            )

    treatment_data[TT_col] = treatment_data[TT_col].fillna(0)

    if after_treatment_period:
        
        treatment_period_last = datetime.strptime(
            treatment_period[1], 
            date_format
            )
        after_treatment_period_day1 = treatment_period_last + timedelta(days=1)
        
        after_treatment_period_range = pd.date_range(
            start = after_treatment_period_day1, 
            end = study_period[1],
            freq = freq
            )
        after_treatment_period_N = len(after_treatment_period_range)

        ATT_dummies = [1] * after_treatment_period_N

        ATT_data = {
            config.TIME_COL: after_treatment_period_range, 
            ATT_col: ATT_dummies
            }
        ATT_data = pd.DataFrame(ATT_data)

        after_treatment_data = study_period_range.merge(ATT_data, how = "left")
        after_treatment_data[ATT_col] = after_treatment_data[ATT_col].fillna(0)
        after_treatment_data = after_treatment_data.drop(columns=[config.TIME_COL, config.TIME_COUNTER_COL])

        treatment_data = pd.concat([treatment_data, after_treatment_data], axis=1)

    else:
        after_treatment_period_N = 0
    
    no_treatments = 1

    treatment_meta = {
        "no_treatments": 1,
        "study_period_start": study_period[0],
        "study_period_end": study_period[1],
        "study_period": study_period_N,
        "frequency": freq,
        "date_format": date_format,
        "pre_post": pre_post
        }

    treatment_config = {
        0:
            {
                "treatment_name": treatment_name,
                "treatment_period_start": treatment_period[0],
                "treatment_period_end": treatment_period[1],
                "treatment_period": treatment_period_N,
                "after_treatment_period": after_treatment_period,
                "after_treatment_period_N": after_treatment_period_N,
                "no_treatments": no_treatments,
                "TT_col": TT_col,
                "ATT_col": ATT_col,
                }
            }

    treatment = DiffTreatment(
        treatment_data, 
        treatment_config,
        treatment_meta,
        timestamp = helper.create_timestamp(function="create_treatment")
        )

    if verbose:
        print("OK")

    return treatment

class DiffData:

    def __init__(
        self,
        did_modeldata,
        diff_groups,
        diff_treatment,
        outcome_col_original,
        unit_time_col_original,
        covariates,
        treatment_cols,
        timestamp
        ):

        self.data = [
            did_modeldata, 
            diff_groups, 
            diff_treatment, 
            outcome_col_original,
            unit_time_col_original,
            covariates,
            treatment_cols,
            timestamp
            ]

    def get_did_modeldata_df (self):
        return pd.DataFrame(self.data[0])

    def get_did_groups(self):
        return self.data[1]

    def get_did_treatment(self):
        return self.data[2]

    def get_unit_time_cols(self):
        return self.data[4]

    def get_covariates(self):
        return self.data[5]
    
    def get_treatment_cols(self):
        return self.data[6]
    
    def get_timestamp(self):
        return self.data[7]

    def add_covariates(
        self, 
        additional_df: pd.DataFrame,
        variables: list = None,
        unit_col: str = None,
        time_col: str = None,
        verbose: bool = config.VERBOSE
        ):
        
        if unit_col is None and time_col is None:
            raise ValueError("Parameter 'unit_col' and/or 'time_col' must be stated")
        
        if verbose:
            if len(variables) > 0:
                print(f"Adding covariates {', '.join(variables)} from additional data frame to model data", end = " ... ")
            else:
                print("Merging additional data frame to model data", end = " ... ")

        did_modeldata = self.get_did_modeldata_df()
        
        additional_df = tools.panel_index(
            data=additional_df,
            unit_col=unit_col,
            time_col=time_col,
            verbose=verbose
            )
        
        existing_variables = []
        
        if unit_col is not None and time_col is not None:            
            
            if variables is None:

                existing_variables = [var for var in additional_df.colums if var in did_modeldata.columns]

                did_modeldata = pd.merge(
                    did_modeldata, 
                    additional_df, 
                    on = config.UNIT_TIME_COL, 
                    how = "inner"
                    )
                
            else:

                existing_variables = [var for var in variables if var in did_modeldata.columns]

                additional_df_cols = [config.UNIT_TIME_COL] + [col for col in additional_df.columns if col in variables]

                did_modeldata = pd.merge(
                    did_modeldata, 
                    additional_df[additional_df_cols], 
                    on = config.UNIT_TIME_COL, 
                    how = "inner"
                    )

        if unit_col is not None and time_col is None:

            if variables is None:
                
                existing_variables = [var for var in additional_df.colums if var in did_modeldata.columns]

                did_modeldata = pd.merge(
                    did_modeldata, 
                    additional_df, 
                    left_on = config.UNIT_COL,
                    right_on = unit_col,
                    how = "inner"
                    )
                
            else:
                
                existing_variables = [var for var in variables if var in did_modeldata.columns]

                additional_df_cols = [unit_col] + [col for col in additional_df.columns if col in variables]

                did_modeldata = pd.merge(
                    did_modeldata, 
                    additional_df[additional_df_cols], 
                    left_on = config.UNIT_COL, 
                    right_on = unit_col,
                    how = "inner"
                    ) 

        if time_col is not None and unit_col is None:

            additional_df_cols = [unit_col] + [col for col in additional_df.columns if col in variables]

            existing_variables = [var for var in additional_df_cols if var in did_modeldata.columns]

            did_modeldata = pd.merge(
                did_modeldata, 
                additional_df[additional_df_cols], 
                left_on = config.UNIT_COL,
                right_on = config.TIME_COL,
                how = "inner"
                )  
        
        self.data[0] = did_modeldata
        self.data[5] = variables

        if verbose:
            print("OK")
        
        if len(existing_variables) > 0:
            print(f"WARNING: Additional data frame includes duplicate column names: {', '.join(existing_variables)}")
        
        return self

    def add_treatment(
        self,
        treatment_name,
        treatment_period,
        treatment_group,
        control_group,
        after_treatment_period: bool = False,
        after_treatment_name = None,
        verbose: bool = config.VERBOSE
        ):

        if not treatment_name:
            
            raise ValueError("When adding a treatment, you need to specify a treatment name with parameter treament_name = [your_treatment].")
        
        else:
            
            treatment_name = tools.clean_treatment_name(treatment_name)

        if verbose:
            print(f"Adding treatment '{treatment_name} to model data", end = " ... ")

        did_treatment_old = self.get_did_treatment()
        treatment_config_old = did_treatment_old.get_config()
        treatment_meta_old = did_treatment_old.get_metadata()
        if treatment_meta_old["pre_post"]:
            raise ValueError ("Adding treatments in a pre-post design is currently not possible.")
        no_treatments_old = treatment_meta_old["no_treatments"]

        did_groups_old = self.get_did_groups()
        groups_config_old = did_groups_old.get_config()
        groups_data_old = did_groups_old.get_data()

        did_modeldata_old = self.get_did_modeldata_df()
        unit_id_col, time_col = self.get_unit_time_cols()
        outcome_col_original = self.data[3]
        unit_time_col_original = self.get_unit_time_cols()
        covariates = self.get_covariates()

        treatment_cols = self.get_treatment_cols()
        treatment_cols_new = treatment_cols

        no_treatments = no_treatments_old+1
        key_counter = no_treatments-1

        new_groups = create_groups(
            treatment_group = treatment_group, 
            control_group = control_group,
            treatment_name = treatment_name,
            verbose=False
            )
        new_groups_data_df = new_groups.get_data()[0]        
        new_groups_config = new_groups.get_config()
        TG_col = new_groups_config[0]["TG_col"]

        new_treatment = create_treatment(
            study_period = [treatment_meta_old["study_period_start"], treatment_meta_old["study_period_end"]],
            treatment_period = treatment_period,
            freq = treatment_meta_old["frequency"],
            date_format = treatment_meta_old["date_format"],
            treatment_name = treatment_name,
            pre_post = treatment_meta_old["pre_post"],
            after_treatment_period = after_treatment_period,
            verbose=False
            )
        new_treatment_data_df = new_treatment.get_data()
        
        new_treatment_config = new_treatment.get_config()        
        TT_col = new_treatment_config[0]["TT_col"]
        ATT_col = new_treatment_config[0]["ATT_col"]

        new_merge = new_groups_data_df.merge(
            new_treatment_data_df, 
            how = "cross"
            )
        
        new_merge[treatment_name] = new_merge[TG_col]*new_merge[TT_col]

        if after_treatment_period:
            new_merge[after_treatment_name] = new_merge[TG_col]*new_merge[ATT_col]
        
        new_merge = tools.panel_index(
            data=new_merge,
            unit_col=config.UNIT_COL,
            time_col=config.TIME_COL,
            verbose=verbose
            )

        did_modeldata_old = tools.panel_index(
            data=did_modeldata_old,
            unit_col=config.UNIT_COL,
            time_col=config.TIME_COL,
            verbose=verbose
            )

        new_merge = new_merge[
            [
                config.UNIT_TIME_COL, 
                TT_col, 
                TG_col, 
                treatment_name
                ]
                ].copy()

        did_modeldata_new = did_modeldata_old.merge(
            new_merge,
            how = "inner",
            left_on = config.UNIT_TIME_COL,
            right_on = config.UNIT_TIME_COL
            )        
        
        treatment_cols_new[key_counter] = {
            "TT_col": TT_col, 
            "ATT_col": ATT_col, 
            "treatment_name": treatment_name, 
            "after_treatment_name": after_treatment_name                
            }
        
        groups_config_new = groups_config_old
        groups_config_new[key_counter] = new_groups_config[0] 
        groups_data_new = groups_data_old
        groups_data_old.append(new_groups_data_df) 
        groups_new = DiffGroups(
            groups_data_new, 
            groups_config_new,
            timestamp = helper.create_timestamp(function="add_treatment")
            )

        treatment_meta_new = treatment_meta_old
        treatment_meta_new["no_treatments"] = no_treatments
        treatment_config_new = treatment_config_old
        treatment_config_new[key_counter] = new_treatment_config[0]
        
        treatment_new = DiffTreatment(
            new_treatment_data_df, 
            treatment_config_new,
            treatment_meta_new,
            timestamp = helper.create_timestamp(function="add_treatment")
            )

        did_data_new = DiffData(
            did_modeldata = did_modeldata_new, 
            diff_groups = groups_new, 
            diff_treatment = treatment_new, 
            outcome_col_original = outcome_col_original,
            unit_time_col_original = unit_time_col_original,
            covariates = covariates,
            treatment_cols = treatment_cols_new,
            timestamp = helper.create_timestamp(function="add_segmentation")
            )

        if verbose:
            print("OK")

        return did_data_new

    def add_segmentation(
        self,
        group_benefit: list
        ):

        diff_groups = self.data[1]
        did_modeldata = self.data[0]

        diff_groups = diff_groups.add_segmentation(group_benefit = group_benefit)

        did_modeldata[config.BG_COL] = 0
        did_modeldata.loc[did_modeldata[config.UNIT_COL].astype(str).isin(group_benefit), config.BG_COL] = 1

        self.data[1] = diff_groups
        self.data[0] = did_modeldata

        return self
    
    def add_own_counterfactual(
        self,
        additional_df,
        counterfactual_outcome_col,
        time_col,
        counterfactual_UID = "counterfac",
        verbose: bool = config.VERBOSE
        ):
        
        if time_col is None or counterfactual_outcome_col is None:
            raise ValueError("Parameters 'time_col' and 'counterfactual_outcome_col' must be stated")
        
        tools.check_columns(
            df = additional_df,
            columns = [counterfactual_outcome_col, time_col]
            )

        did_modeldata = self.data[0]
        groups_data = self.data[1].get_data()
        groups_config = self.data[1].get_config()
        treatment_group = groups_data.loc[groups_data[config.TG_COL] == 1, config.UNIT_COL].values
        treatment_config = self.data[2].get_config()        
        outcome_col_original = self.data[3]        
        treatment_data = self.data[2].get_data()

        additional_df = additional_df[[time_col, counterfactual_outcome_col]].copy()
        
        did_modeldata_TG = did_modeldata[did_modeldata[config.UNIT_COL].astype(str).isin(treatment_group)].copy()        
        
        did_modeldata_counterfac = pd.DataFrame(columns=did_modeldata_TG.columns, index=range(len(treatment_data)))
        did_modeldata_counterfac[config.UNIT_COL] = counterfactual_UID
        did_modeldata_counterfac[config.TG_COL] = 0
        did_modeldata_counterfac[config.TIME_COL] = treatment_data[config.TIME_COL].values
        did_modeldata_counterfac[config.TIME_COUNTER_COL] = treatment_data[config.TIME_COUNTER_COL].values
        did_modeldata_counterfac[config.TT_COL] = treatment_data[config.TT_COL].values
        did_modeldata_counterfac[config.TREATMENT_COL] = did_modeldata_counterfac[config.TG_COL] * did_modeldata_counterfac[config.TT_COL]

        did_modeldata_counterfac = tools.panel_index(
            data=did_modeldata_counterfac,
            unit_col=config.UNIT_COL,
            time_col=config.TIME_COL,
            verbose=verbose
            )
        
        if treatment_config["after_treatment_period"]:
            did_modeldata_counterfac["ATT"] = treatment_data["ATT"].values

        if counterfactual_outcome_col == outcome_col_original:
            additional_df = additional_df.rename(columns={counterfactual_outcome_col: counterfactual_outcome_col+"_cf"})
            counterfactual_outcome_col = counterfactual_outcome_col+"_cf"
        
        did_modeldata_counterfac = pd.merge(
            did_modeldata_counterfac,
            additional_df,
            left_on = config.TIME_COL,
            right_on = time_col,
            how = "left"
            )

        did_modeldata_counterfac[outcome_col_original] = did_modeldata_counterfac[counterfactual_outcome_col]        
        did_modeldata_counterfac = did_modeldata_counterfac.drop(counterfactual_outcome_col, axis = 1) 

        did_modeldata_TG_with_counterfac = pd.concat(
            [did_modeldata_TG, did_modeldata_counterfac], 
            ignore_index=True
            )       
     
        groups_config["counterfactual"] = True
        
        groups_data = groups_data[groups_data[config.TG_COL] == 1]
        groups_data_cf = {
            config.UNIT_COL: counterfactual_UID, 
            config.TG_COL: 0
            }
        groups_data = pd.concat([groups_data, pd.DataFrame([groups_data_cf])], ignore_index=True)
        
        groups = DiffGroups(
            groups_data, 
            groups_config,
            timestamp = helper.create_timestamp(function="add_own_counterfactual")
            )
        
        self.data[0] = did_modeldata_TG_with_counterfac
        self.data[1] = groups
        
        return self

    def summary(self):

        did_modeldata = self.data[0]
       
        groups_config = self.data[1].get_config()

        treatment_config = self.data[2].get_config()
        treatment_meta = self.data[2].get_metadata()
        
        outcome_col_original = self.data[3]

        title = f"{config.DID_DESCRIPTION} {config.TREATMENT_DESCRIPTION}s and {config.GROUP_DESCRIPTION}s"
        total_width = len(title)

        print("=" * total_width)
        print(title)
        print("-" * total_width)

        for key, value in treatment_config.items():
            
            if value["treatment_name"] is not None:
                print (f"{config.TREATMENT_DESCRIPTION} {value['treatment_name']}")
            else:
                print (f"{config.TREATMENT_DESCRIPTION} {key+1}")
            
            print(f" {config.GROUPS_DESCRIPTION}")
            print(f"  {config.UNITS_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {groups_config[key]['full_sample']} ({round(groups_config[key]['full_sample']/groups_config[key]['full_sample']*100, config.ROUND_PERCENT)} %)")
            print(f"  {config.TREATMENT_GROUP_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {groups_config[key]['treatment_group']} ({round(groups_config[key]['treatment_group']/groups_config[key]['full_sample']*100, config.ROUND_PERCENT)} %)")
            print(f"  {config.CONTROL_GROUP_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {groups_config[key]['control_group']} ({round(groups_config[key]['control_group']/groups_config[key]['full_sample']*100, config.ROUND_PERCENT)} %)")
            print(f"  {config.DDD_GROUP_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {tools.bool_to_YN(groups_config[key]['DDD'])}")
            
            print(f" {config.TIME_PERIODS_DESCRIPTION}")
            if treatment_meta["pre_post"]:
                print(f"  {config.STUDY_PERIOD_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {treatment_config[key]['treatment_period_start']} vs. {treatment_config[key]['treatment_period_end']} ({config.PREPOST_DESCRIPTION})")
            else:
                print(f"  {config.STUDY_PERIOD_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {treatment_meta['study_period_start']} - {treatment_meta['study_period_end']} ({treatment_meta['study_period']} {treatment_meta['frequency']})")
                print(f"  {config.TREATMENT_PERIOD_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}} {treatment_config[key]['treatment_period_start']} - {treatment_config[key]['treatment_period_end']} ({treatment_config[key]['treatment_period']} {treatment_meta['frequency']})")
                
        print("-" * total_width)

        print(f"Outcome '{outcome_col_original}'")
        print(f"{'Mean':<{config.DIDDATA_SUMMARY_MAX_WIDTH}}   {round(np.mean(did_modeldata[outcome_col_original]), config.ROUND_STATISTIC)}")
        print(f"{'Standard deviation':<{config.DIDDATA_SUMMARY_MAX_WIDTH}}   {round(np.std(did_modeldata[outcome_col_original]), config.ROUND_STATISTIC)}")
        print(f"{config.N_DESCRIPTION:<{config.DIDDATA_SUMMARY_MAX_WIDTH}}   {len(did_modeldata)}")

        print("=" * total_width)

    def analysis(
        self, 
        log_outcome: bool = False, 
        FE_unit: bool = False, 
        FE_time: bool = False,
        cluster_SE_by: str = None,
        intercept: bool = True, 
        ITE: bool = False,
        GTE: bool = False,
        ITT: bool = False,
        GTT: bool = False,
        group_by: str = None,
        spillover_treatment: list = [],
        spillover_units: list = [],
        confint_alpha = 0.05,
        bonferroni: bool = False,
        drop_missing: bool = True,
        missing_replace_by_zero: bool = False,
        verbose: bool = config.VERBOSE
        ):

        did_modeldata = self.get_did_modeldata_df()
        outcome_col_original = self.data[3]

        covariates = self.data[5]

        groups_config = self.data[1].get_config()       
        
        treatment_config = self.data[2].get_config()
        
        treatment_meta = self.data[2].get_metadata()        
        freq = treatment_meta["frequency"]
        date_format = treatment_meta["date_format"]
        
        if groups_config[0]["DDD"]:            
                    
            TG_col = groups_config[0]["TG_col"]
            
            treatment_config = treatment_config[0]
            TT_col = treatment_config["TT_col"]
                        
            ddd_results = didanalysis.ddd_analysis(
                data = did_modeldata,
                unit_col = config.UNIT_COL,
                time_col = config.TIME_COL,
                TG_col = TG_col,
                TT_col = TT_col,
                BG_col = config.BG_COL,
                outcome_col = outcome_col_original,
                log_outcome = log_outcome,
                log_outcome_add = 0.01,
                FE_unit = FE_unit,
                FE_time = FE_time,                
                covariates = covariates,
                confint_alpha = confint_alpha,
                freq = freq,
                date_format = date_format,
                drop_missing = drop_missing,
                missing_replace_by_zero = missing_replace_by_zero,
                verbose = verbose
                )

            return ddd_results
        
        else:           
        
            treatment_cols = self.get_treatment_cols()
            
            TT_col = [None]*len(treatment_cols)
            TG_col = [None]*len(treatment_cols)
            treatment_col = [None]*len(treatment_cols)
            after_treatment_col = [None]*len(treatment_cols)
            ATT_col = [None]*len(treatment_cols)
            
            for key, value in treatment_cols.items():

                TG_col[key] = groups_config[key]["TG_col"]
                TT_col[key] = value["TT_col"]
                treatment_col[key] = value["treatment_name"]
                
                if value["after_treatment_name"] is not None:
                    after_treatment_col[key] = value["after_treatment_name"]
                if value["ATT_col"] is not None:
                    ATT_col[key] = value["ATT_col"]
                    
            did_results = didanalysis.did_analysis(
                data = did_modeldata,
                TG_col = TG_col,
                TT_col = TT_col,
                treatment_col = treatment_col,
                unit_col = config.UNIT_COL,
                time_col = config.TIME_COL,
                outcome_col = outcome_col_original,
                after_treatment_col = after_treatment_col,
                ATT_col = ATT_col,
                pre_post = treatment_meta["pre_post"],
                log_outcome = log_outcome,
                FE_unit = FE_unit,
                FE_time = FE_time,
                cluster_SE_by = cluster_SE_by,
                intercept = intercept,
                ITE = ITE,
                GTE = GTE,
                ITT = ITT,
                GTT = GTT,
                group_by = group_by,
                covariates = covariates,
                spillover_treatment = spillover_treatment,
                spillover_units = spillover_units,
                confint_alpha = confint_alpha,
                bonferroni = bonferroni,
                freq = freq,
                date_format = date_format,
                drop_missing = drop_missing,
                missing_replace_by_zero = missing_replace_by_zero,
                verbose = verbose
                )

            return did_results

def merge_data(
    outcome_data,
    unit_id_col,
    time_col,
    outcome_col,
    diff_groups,
    diff_treatment,
    drop_missing: bool = True,
    missing_replace_by_zero: bool = False,
    keep_columns: bool = False,
    verbose: bool = config.VERBOSE
    ):
    
    tools.check_columns(
        df = outcome_data,
        columns = [
            unit_id_col, 
            time_col, 
            outcome_col
            ]
        )

    if verbose:
        print("Merging groups and treatment data", end = " ... ")

    groups_data_df = diff_groups.get_data()
    groups_data_df = groups_data_df[0]
    
    groups_config = diff_groups.get_config()
    groups_config = groups_config[0]
    
    treatment_data_df = diff_treatment.get_data()
    treatment_config = diff_treatment.get_config()
    treatment_config = treatment_config[0]    

    TT_col = treatment_config["TT_col"]
    
    if treatment_config["after_treatment_period"]:
        ATT_col = treatment_config["ATT_col"]
    else:
        ATT_col = None
        after_treatment_name = None
    
    TG_col = groups_config["TG_col"]
        
    if treatment_config["treatment_name"] is not None:
        treatment_name = treatment_config["treatment_name"]        
    else:
        treatment_name = config.TREATMENT_COL
        
    if treatment_config["after_treatment_period"]:
        if treatment_config["treatment_name"] is not None:
            after_treatment_name = treatment_name + "_after"
        else:
            after_treatment_name = f"{config.TG_COL}{config.DELIMITER_INTERACT}{config.ATT_COL}"
      
    did_modeldata = groups_data_df.merge(
        treatment_data_df, 
        how = "cross"
        )

    did_modeldata[treatment_name] = did_modeldata[TG_col] * did_modeldata[TT_col]

    if treatment_config["after_treatment_period"]:
        did_modeldata[after_treatment_name] = did_modeldata[TG_col] * did_modeldata[ATT_col]

    did_modeldata = tools.panel_index(
        data=did_modeldata,
        unit_col=config.UNIT_COL,
        time_col=config.TIME_COL,
        verbose=verbose
        )

    outcome_data = tools.panel_index(
        data=outcome_data,
        unit_col=unit_id_col,
        time_col=time_col,
        verbose=verbose
        )

    if keep_columns:
        outcome_data_short = outcome_data
    else:
        outcome_data_short = outcome_data[[config.UNIT_TIME_COL, outcome_col]]

    did_modeldata = did_modeldata.merge(
        outcome_data_short, 
        on=config.UNIT_TIME_COL, 
        how="left"
        )
    
    if drop_missing or missing_replace_by_zero:
        modeldata_ismissing = tools.is_missing(
            data = did_modeldata, 
            drop_missing = drop_missing,
            missing_replace_by_zero = missing_replace_by_zero,
            verbose = False
            )
        did_modeldata = modeldata_ismissing[2]
 
    outcome_col_original = outcome_col
    unit_time_col_original = unit_id_col, time_col
    
    treatment_cols = {
        0:  {
            "TT_col": TT_col, 
            "ATT_col": ATT_col, 
            "TG_col": TG_col,
            "treatment_name": treatment_name, 
            "after_treatment_name": after_treatment_name
            }
        }

    timestamp = helper.create_timestamp(function="merge_data")

    did_data_all = DiffData(
        did_modeldata, 
        diff_groups, 
        diff_treatment, 
        outcome_col_original,
        unit_time_col_original,
        [],
        treatment_cols,
        timestamp
        )

    if verbose:
        print("OK")

    return did_data_all

def create_data(
    outcome_data,
    unit_id_col,
    time_col,
    outcome_col,
    treatment_group,
    control_group,
    study_period,
    treatment_period,
    treatment_name: str = None,
    freq = "D",
    date_format = "%Y-%m-%d",
    pre_post: bool = False,
    after_treatment_period: bool = False,
    drop_missing: bool = True,
    missing_replace_by_zero: bool = False,
    verbose: bool = config.VERBOSE
    ):

    groups = create_groups(
        treatment_group, 
        control_group,
        treatment_name = treatment_name,
        verbose = verbose
        )
    
    treatment = create_treatment(
        study_period = study_period, 
        treatment_period = treatment_period, 
        freq = freq,
        date_format = date_format,
        pre_post = pre_post,
        treatment_name = treatment_name,
        after_treatment_period = after_treatment_period,
        verbose = verbose
        )    
    
    did_data_all = merge_data(
        outcome_data = outcome_data,
        unit_id_col = unit_id_col,
        time_col = time_col,
        outcome_col = outcome_col,
        diff_groups = groups,
        diff_treatment = treatment,
        drop_missing = drop_missing,
        missing_replace_by_zero = missing_replace_by_zero,
        verbose = verbose
        )

    did_data_all.timestamp = helper.create_timestamp(function="create_data")

    return did_data_all

def create_counterfactual(
    data,
    y: str,
    X: list,
    unit_col: str,
    treatment_col: str,
    time_col: str,
    cf_for_unit: str,
    use_data: str = "both",
    model_type: str = "ols",
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
            
    data = data[[y] + X + [unit_col, treatment_col, time_col]].copy()

    data_len = len(data)
    data = data.dropna()
    if len(data) < data_len:
        print(f"NOTE: Because of NaN values, {data_len-len(data)} observations were skipped.")

    data = data[data[unit_col].astype(str) != cf_for_unit]
    data_unit = data[data[unit_col].astype(str) == cf_for_unit]
    
    isnotreatment = tools.is_notreatment(
        data = data,
        unit_col = unit_col,
        treatment_col = treatment_col
        )
    control_group = isnotreatment[2]
    
    units_tt = tools.treatment_times(
        data = data,
        unit_col = unit_col,
        time_col = time_col,
        treatment_col = treatment_col
        )[0]
    units = tools.unique(units_tt[unit_col])  
    
    if not isnotreatment[0]:
        print(f"NOTE: No {config.NO_TREATMENT_CG_DESCRIPTION}. Counterfactual will not cover full treatment time.")    
    
    data_TG = pd.DataFrame(columns = data.columns)

    for unit in units:
        data_TG_unit = data.loc[data[unit_col].astype(str) == unit]
        data_TG_unit = data_TG_unit[data_TG_unit[time_col] < units_tt.loc[unit_col == unit, "treatment_min"]]
        data_TG = pd.concat(
            [data_TG, data_TG_unit],
            ignore_index=True
        )
    data_CG = data[data[unit_col].astype(str).isin(control_group)].copy()
     
    if use_data == "treatment":
        data_cf = data_TG       
    elif use_data == "control":
        data_cf = data_CG
    else:
        data_cf = pd.concat(
            [
                data_TG, 
                data_CG
                ],
            ignore_index=True
        )
    
    data_cf[X] = data_cf[X].apply(pd.to_numeric, errors='coerce')
    
    counterfactual_pred = tools.model_wrapper(
        y = data_cf[y],
        X = data_cf[X],
        model_type = model_type,
        test_size = test_size,
        train_size = train_size,
        model_n_estimators = model_n_estimators,
        model_max_features = model_max_features,
        model_min_samples_split = model_min_samples_split,
        rf_max_depth = rf_max_depth,
        gb_iterations = gb_iterations,
        gb_max_depth = gb_max_depth,
        gb_learning_rate = gb_learning_rate,
        knn_n_neighbors = knn_n_neighbors,
        svr_kernel = svr_kernel,
        xgb_learning_rate = xgb_learning_rate,
        lgbm_learning_rate = lgbm_learning_rate,
        random_state = random_state
        )
    
    return [
        counterfactual_pred, 
        data_cf, 
        data_unit
        ]