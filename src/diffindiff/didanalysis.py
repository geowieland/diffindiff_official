#-----------------------------------------------------------------------
# Name:        didanalysis (diffindiff package)
# Purpose:     Analysis functions for difference-in-differences analyses
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     2.3.0
# Last update: 2026-03-01 11:23
# Copyright (c) 2024-2026 Thomas Wieland
#-----------------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import diffindiff.didtools as tools
import diffindiff.config as config
import diffindiff.didanalysis_helper as helper

class DiffModel:

    def __init__(
        self,
        did_modelresults,
        did_modelconfig,
        did_modeldata,
        did_modelpredictions,
        did_model,
        timestamp
        ):

        """
        Container class for DiD model results.

        Parameters
        ----------
        did_modelresults : list
            List containing the model results (coefficients, fixed effects, etc.).
        did_modelconfig : dict
            Configuration dictionary for the DiD model.
        did_modeldata : pd.DataFrame
            Panel data used for the estimation of the DiD model.
        did_modelpredictions : pd.DataFrame
            Predicted values of the DiD model.
        did_model: statsmodels regression results object
            The fitted DiD model object from statsmodels.
        
        Returns
        -------
        None
            Constructor does not return a value; instance is initialized in-place.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model.summary()
        """

        self.data = [
            did_modelresults, 
            did_modelconfig, 
            did_modeldata, 
            did_modelpredictions, 
            did_model,
            timestamp
            ]    

    def get_did_modeldata_df (self):
        
        """
        Return the model data as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame used for model estimation.

        Examples
        --------
        >>> curfew_data_prepost=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D',
        ...     pre_post=True
        ... )
        >>> model=curfew_data_prepost.analysis()
        >>> modeldata_df=model.get_did_modeldata_df()
        >>> print(modeldata_df.head())
        """

        return pd.DataFrame(self.data[2])
    
    def treatment_statistics(
        self,
        treatment: str = None,
        after_treatment_col: str = None
        ):

        """
        Compute summary statistics about the treatment and study period.

        Parameters
        ----------
        treatment : str, optional
            Treatment name to analyse. If None, the first treatment in the model config is chosen.
        after_treatment_col : str, optional
            Column name of an after-treatment indicator, if present.

        Returns
        -------
        list
            [group_sizes, average_treatment_time, groups, treatment_timepoints, time_periods]

        Raises
        ------
        ValueError
            If a requested treatment is not present in the model object.

        Examples
        --------
        >>> curfew_data_prepost=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D',
        ...     pre_post=True
        ... )
        >>> model=curfew_data_prepost.analysis()
        >>> model_treatment_statistics = model.treatment_statistics()
        >>> print(model_treatment_statistics)
        """

        model_config = self.data[1]
        model_data = self.data[2]
               
        if treatment is not None:
            
            if treatment not in model_config["treatment_col"]:
                raise ValueError(f"Treatment {treatment} not in model object.")
        else:
            
            treatment = model_config["treatment_col"][0]
            print(f"NOTE: No treatment was stated. Choosing treatment {treatment} for analysis.")

        if after_treatment_col is not None:
            
            if len(model_config["after_treatment_col"]) == 0:
                raise ValueError("Model object does not include an after-treatment period.")
            
            if after_treatment_col not in model_config["after_treatment_col"]:
                raise ValueError(f"Treatment {treatment} not in model object.")

        unit_col = model_config["unit_col"]
        time_col = model_config["time_col"]

        treatment_timepoints = model_data.groupby(unit_col)[treatment].sum()
        treatment_timepoints = pd.DataFrame(treatment_timepoints)
        treatment_timepoints = treatment_timepoints.reset_index()

        study_period_start = pd.to_datetime(min(model_data[time_col]))
        study_period_start = study_period_start.date()
        study_period_end = pd.to_datetime(max(model_data[time_col]))
        study_period_end = study_period_end.date()
        study_period_N = model_data[time_col].nunique()
        
        if len(model_data[model_data[treatment] == 1]) > 0:
            treatment_period_start = pd.to_datetime(min(model_data[model_data[treatment] == 1][time_col]))
            treatment_period_end = pd.to_datetime(max(model_data[model_data[treatment] == 1][time_col]))
            treatment_period_N = model_data.loc[model_data[treatment] == 1, time_col].nunique()
        else:
            treatment_period_N = 0
            
        after_treatment_period_start = None
        after_treatment_period_end = None
        after_treatment_period_N = None
        if len(model_config["after_treatment_col"]) > 0 and after_treatment_col is not None:
            after_treatment_period_start = treatment_period_end+pd.Timedelta(days=1)
            after_treatment_period_start = pd.to_datetime(after_treatment_period_start)
            after_treatment_period_end = pd.to_datetime(study_period_end)
            after_treatment_period_N = model_data.loc[model_data[after_treatment_col] == 1, time_col].nunique()            
            after_treatment_period_start = after_treatment_period_start.strftime(model_config["date_format"])
            after_treatment_period_end = after_treatment_period_end.strftime(model_config["date_format"])
            
        study_period_start = study_period_start.strftime(model_config["date_format"])
        study_period_end = study_period_end.strftime(model_config["date_format"])
        if treatment_period_N > 0:
            treatment_period_start = treatment_period_start.strftime(model_config["date_format"])
            treatment_period_end = treatment_period_end.strftime(model_config["date_format"])
        else:
            treatment_period_start = None
            treatment_period_end = None
        period_study = [study_period_start, study_period_end, study_period_N]
        period_treatment = [treatment_period_start, treatment_period_end, treatment_period_N]
        period_after_treatment = [after_treatment_period_start, after_treatment_period_end, after_treatment_period_N]
        time_periods = [period_study, period_treatment, period_after_treatment]

        treatment_group = np.array(treatment_timepoints[treatment_timepoints[treatment] > 0][unit_col])
        control_group = np.array(treatment_timepoints[treatment_timepoints[treatment] == 0][unit_col])
        groups = [treatment_group, control_group]

        treatment_group_size = len(treatment_group)
        control_group_size = len(control_group)
        all_units = treatment_group_size+control_group_size
        treatment_group_share = treatment_group_size/all_units
        control_group_share = control_group_size/all_units
        group_sizes = [treatment_group_size, control_group_size, all_units, treatment_group_share, control_group_share]

        if treatment_period_N > 0:
            average_treatment_time = treatment_timepoints[treatment_timepoints[unit_col].isin(treatment_group)][treatment].mean()
        else:
            average_treatment_time = 0

        return [
            group_sizes,
            average_treatment_time, 
            groups, 
            treatment_timepoints, 
            time_periods
            ]
    
    def treatment_diagnostics(self):

        """
        Assemble treatment diagnostics stored in the model configuration.

        Returns
        -------
        list
            [treatment_diagnostics_df, no_control_conditions]

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model_treatment_diagnostics = model.treatment_diagnostics()
        >>> print(model_treatment_diagnostics[0])
        """

        model_config = self.data[1]
        treatment_diagnostics = model_config["treatment_diagnostics"]
        
        treatment_diagnostics_df = pd.DataFrame()
        treatment_diagnostics_rows = []
        no_control_conditions = []

        for treatment_key, treatment_value in treatment_diagnostics.items():

            treatment_row = {}
            
            if not treatment_value["is_notreatment"] and treatment_value["adoption_type"] == config.TREATMENT_SIMULTANEOUS_DESCRIPTION:
                no_control_conditions.append(treatment_value["treatment"])                                
            
            for treatment_diagnostic_key, treatment_diagnostic_value in treatment_value.items():            

                diagnostic_config = config.TREATMENT_DIAGNOSTICS[treatment_diagnostic_key]

                if diagnostic_config["show_in_summary"]:
                    treatment_row[diagnostic_config["description"]] = tools.bool_to_YN(treatment_diagnostic_value)
                
            treatment_diagnostics_rows.append(treatment_row)
  
        treatment_diagnostics_df = pd.DataFrame(treatment_diagnostics_rows)            
        treatment_diagnostics_df = treatment_diagnostics_df.reset_index(drop=True)
        
        return [
            treatment_diagnostics_df, 
            no_control_conditions
            ]
    
    def data_diagnostics(self):

        """
        Return assembled data diagnostics from the model configuration.

        Returns
        -------
        pandas.DataFrame
            Table with data diagnostic indicators.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model_data_diagnostics = model.data_diagnostics()
        >>> print(model_data_diagnostics)
        """

        model_config = self.data[1]
        data_diagnostics = model_config["data_diagnostics"]
        
        data_diagnostics_rows = []

        for key, value in data_diagnostics.items():
            
            data_diagnostic_config = config.DATA_DIAGNOSTICS[key]

            if data_diagnostic_config["show_in_summary"]:
                
                description = config.DATA_DIAGNOSTICS[key]["description"] 

                data_diagnostics_rows.append(
                    {
                        config.DATA_DIAGNOSTICS_DESCRIPTION: description, 
                        config.DIAGNOSTICS_COLUMN: tools.bool_to_YN(value)
                        }
                        )

        data_diagnostics_df = pd.DataFrame(data_diagnostics_rows)            
        data_diagnostics_df = data_diagnostics_df.reset_index(drop=True)
           
        return data_diagnostics_df   
    
    def fit_metrics(self):

        """
        Compute and format fit metrics for the model predictions.

        Returns
        -------
        pandas.DataFrame
            Formatted DataFrame with selected fit metrics.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D',
        ...     pre_post=True
        ... )
        >>> model=curfew_data.analysis()
        >>> model_fit_metrics = model.fit_metrics()
        >>> print(model_fit_metrics)
        """

        data = self.data[2]
        
        model_predictions = self.data[3]

        model_config = self.data[1]
        outcome_col = model_config["outcome_col"]
        indep_vars_no = model_config["indep_vars_no"]

        model_fit_metrics = helper.fit_metrics(
            data,
            outcome_col,
            model_predictions,
            indep_vars_no = indep_vars_no
            )[1]        
       
        model_fit_metrics_rows = []

        for key, value in model_fit_metrics.items():

            model_fit_metrics_config = config.MODEL_FIT_METRICS[key]

            if model_fit_metrics_config["show_in_summary"]:
                
                description = config.MODEL_FIT_METRICS[key]["description"] 

                model_fit_metrics_rows.append(
                    {
                        config.MODEL_FIT_METRICS_DESCRIPTION: description, 
                        config.DIAGNOSTICS_COLUMN: value
                        }
                        )

        model_fit_metrics_df = pd.DataFrame(model_fit_metrics_rows)            
        model_fit_metrics_df = model_fit_metrics_df.reset_index(drop=True)
           
        return model_fit_metrics_df   

    def treatment_effects(self):

        """
        Extract and format treatment effect estimates from model results.

        Returns
        -------
        pandas.DataFrame
            Estimates with confidence intervals and p-values formatted for display.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model_treatment_effects = model.treatment_effects()
        >>> print(model_treatment_effects)
        """

        model_results = self.data[0]
      
        treatment_effects_df = pd.DataFrame()
        
        for effect in config.EFFECTS_TYPES_MODEL_RESULTS_SUMMARY:            

            if effect in model_results:               
             
                description = None
                summary_description = None                
                
                for effects_key, effects_value in config.EFFECTS_TYPES.items():
                    
                    if "model_results_key" in effects_value and effects_value["model_results_key"] == effect:
                        
                        description = effects_value["description"]
                        summary_description = effects_value["summary_description"]
                        
                        break

                effect_estimates = model_results[effect]
                
                effect_estimates_rows = []
            
                for key, value in effect_estimates.items():
                    
                    if isinstance(value, dict):                        
                           
                        summary_text_effect = summary_description.format(
                            description=description,
                            coef=value.get("Coefficient","")
                        )                        

                        effect_estimates_rows.append(
                            {
                                "": summary_text_effect,
                                config.OLS_MODEL_RESULTS["coef"]["summary_description"]: round(value[config.OLS_MODEL_RESULTS["coef"]["model_results_key"]], config.ROUND_STATISTIC),
                                config.OLS_MODEL_RESULTS["coef_standard_errors"]["summary_description"]: round(value[config.OLS_MODEL_RESULTS["coef_standard_errors"]["model_results_key"]], config.ROUND_STATISTIC),
                                config.OLS_MODEL_RESULTS["coef_teststatistic"]["summary_description"]: round(value[config.OLS_MODEL_RESULTS["coef_teststatistic"]["model_results_key"]], config.ROUND_STATISTIC),
                                config.OLS_MODEL_RESULTS["coef_p"]["summary_description"]: round(value[config.OLS_MODEL_RESULTS["coef_p"]["model_results_key"]], config.ROUND_STATISTIC),
                                config.OLS_MODEL_RESULTS["coef_confint_lower"]["summary_description"]: round(value[config.OLS_MODEL_RESULTS["coef_confint_lower"]["model_results_key"]], config.ROUND_STATISTIC),
                                config.OLS_MODEL_RESULTS["coef_confint_upper"]["summary_description"]: round(value[config.OLS_MODEL_RESULTS["coef_confint_upper"]["model_results_key"]], config.ROUND_STATISTIC)
                                }
                            )

                effect_estimates_rows_df = pd.DataFrame(effect_estimates_rows)
                treatment_effects_df = pd.concat(
                    [
                        treatment_effects_df, 
                        effect_estimates_rows_df
                        ], 
                        ignore_index=True
                        )

        treatment_effects_df = treatment_effects_df.reset_index(drop=True)

        return treatment_effects_df

    def covariates(self):

        """
        Return a DataFrame of covariate effect estimates if present.

        Returns
        -------
        pandas.DataFrame or None
            Covariates table or None if no covariates are present.

        Examples
        --------
        >>> curfew_data_prepost=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D',
        ...     pre_post=True
        ... )
        >>> model=curfew_data_prepost.analysis()
        >>> model_covariates = model.covariates()
        >>> print(model_covariates)
        """

        model_results = self.data[0]
       
        if "covariates_effects" in model_results:

            covariates_effects = model_results["covariates_effects"]                            

            covariates_effects_rows = []

            for key, value in covariates_effects.items():
                covariates_effects_rows.append({
                    "": value["Coefficient"],
                    "Estimate": value[config.OLS_MODEL_RESULTS["coef"]["model_results_key"]],
                    "SE": value["SE"],
                    "t": value["t"],
                    "p": value["p"],
                    "CI lower": value["CI_lower"],
                    "CI upper": value["CI_upper"]
                })
            
            covariates_effects_df = pd.DataFrame(covariates_effects_rows)
            covariates_effects_df = covariates_effects_df.reset_index(drop=True)

            return covariates_effects_df
        
        else:

            print("Model does not include covariates.")

            return None
     
    def fixed_effects(
        self,
        units: bool = True,
        time: bool = True,
        group: bool = True
        ):

        """
        Return fixed effects tables for units, time and group as requested.

        Parameters
        ----------
        units : bool, optional
            Include unit fixed effects table. Default is True.
        time : bool, optional
            Include time fixed effects table. Default is True.
        group : bool, optional
            Include group fixed effects table. Default is True.

        Returns
        -------
        list
            [unit_FE_df or None, time_FE_df or None, group_FE_df or None]

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model_fixed_effects = model.fixed_effects()
        >>> print(model_fixed_effects)
        """

        model_results = self.data[0]

        fixed_effects = [None, None, None]
        
        if model_results["fixed_effects"][0] is not None:
            
            fixed_effects_unit = model_results["fixed_effects"][0]["FE_unit"]                            

            fixed_effects_unit_rows = []

            for key, value in fixed_effects_unit.items():
                fixed_effects_unit_rows.append({
                    "Unit": value["Coefficient"],
                    "Estimate": value[config.OLS_MODEL_RESULTS["coef"]["model_results_key"]],
                    "SE": value["SE"],
                    "t": value["t"],
                    "p": value["p"],
                    "CI lower": value["CI_lower"],
                    "CI upper": value["CI_upper"]
                })
            fixed_effects_units_df = pd.DataFrame(fixed_effects_unit_rows)
            fixed_effects_units_df = fixed_effects_units_df.reset_index(drop=True)

            if units:
                fixed_effects[0] = fixed_effects_units_df
        
        else:
            fixed_effects_units_df = None
                        
        if model_results["fixed_effects"][1] is not None:
            
            fixed_effects_time = model_results["fixed_effects"][1]["FE_time"]                         
        
            fixed_effects_time_rows = []

            for key, value in fixed_effects_time.items():
                fixed_effects_time_rows.append({
                    "Time": value["Coefficient"],
                    "Estimate": value[config.OLS_MODEL_RESULTS["coef"]["model_results_key"]],
                    "SE": value["SE"],
                    "t": value["t"],
                    "p": value["p"],
                    "CI lower": value["CI_lower"],
                    "CI upper": value["CI_upper"]
                })
            fixed_effects_time_df = pd.DataFrame(fixed_effects_time_rows)
            fixed_effects_time_df = fixed_effects_time_df.reset_index(drop=True)

            if time:
                fixed_effects[1] = fixed_effects_time_df

        else:
            fixed_effects_time_df = None
            
        if model_results["fixed_effects"][2] is not None:
            
            fixed_effects_group = model_results["fixed_effects"][2]["FE_group"]
        
            fixed_effects_group_rows = []

            for key, value in fixed_effects_group.items():
                fixed_effects_group_rows.append({
                    "Group": value["Coefficient"],
                    "Estimate": value[config.OLS_MODEL_RESULTS["coef"]["model_results_key"]],
                    "SE": value["SE"],
                    "t": value["t"],
                    "p": value["p"],
                    "CI lower": value["CI_lower"],
                    "CI upper": value["CI_upper"]
                })
            fixed_effects_group_df = pd.DataFrame(fixed_effects_group_rows)
            fixed_effects_group_df = fixed_effects_group_df.reset_index(drop=True)

            if group:
                fixed_effects[2] = fixed_effects_group_df

        else:
            fixed_effects_group_df = None

        return fixed_effects
        
    def summary(self):

        """
        Print a readable summary of the DiD model including effects and diagnostics.

        Returns
        -------
        DiffModel
            Returns self to allow method chaining.

        Examples
        --------
        >>> curfew_data_prepost=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D',
        ...     pre_post=True
        ... )
        >>> model=curfew_data_prepost.analysis()
        >>> model.summary()
        """

        model_config = self.data[1]
        no_covariates = len(model_config["covariates"])
        

        treatment_effects_df = self.treatment_effects()        
        
        width = treatment_effects_df[""].str.len().max()
        treatment_effects_df[""] = treatment_effects_df[""].str.ljust(width)
        total_width = (sum(treatment_effects_df.astype(str).map(len).max()) + len(treatment_effects_df.columns) * 2)
        
        print("=" * total_width)
        print(model_config["analysis_description"]) 
        print("-" * total_width)

        print(config.TREATMENT_EFFECTS_DESCRIPTION)
        print(treatment_effects_df.to_string(index=False))        
        print("-" * total_width)

        covariates_effects_df = pd.DataFrame(
            [
                [config.EFFECTS_TYPES["covariates"]["description"], 'YES' if no_covariates > 0 else 'NO'],
                [config.EFFECTS_TYPES["FE"]["types"][0]["description"], tools.bool_to_YN(model_config[config.EFFECTS_TYPES["FE"]["types"][0]["model_config_key"]])],
                [config.EFFECTS_TYPES["FE"]["types"][1]["description"], tools.bool_to_YN(model_config[config.EFFECTS_TYPES["FE"]["types"][1]["model_config_key"]])],
                [config.EFFECTS_TYPES["FE"]["types"][2]["description"], tools.bool_to_YN(model_config[config.EFFECTS_TYPES["FE"]["types"][2]["model_config_key"]])],
                [config.EFFECTS_TYPES["ITT"]["description"], tools.bool_to_YN(model_config[config.EFFECTS_TYPES["ITT"]["model_config_key"]])],
                [config.EFFECTS_TYPES["GTT"]["description"], tools.bool_to_YN(model_config[config.EFFECTS_TYPES["GTT"]["model_config_key"]])],
                ], 
            columns=[config.COVARIATES_DESCRIPTION, "Included"])

        width = covariates_effects_df[config.COVARIATES_DESCRIPTION].str.len().max()
        covariates_effects_df[config.COVARIATES_DESCRIPTION] = covariates_effects_df[config.COVARIATES_DESCRIPTION].str.ljust(width)

        print(covariates_effects_df.to_string(index=False, header=False))
        print("-" * total_width)        

        treatment_diagnostics = self.treatment_diagnostics()
        treatment_diagnostics_df = treatment_diagnostics[0]
        no_control_conditions = treatment_diagnostics[1]       
        
        treatment_diagnostics_df_t = pd.DataFrame(
            treatment_diagnostics_df.values.T, 
            columns = treatment_diagnostics_df["Treatment"].values,
            index = treatment_diagnostics_df.columns)
        treatment_diagnostics_df_t = treatment_diagnostics_df_t.iloc[1:]

        print(config.TREATMENT_DIAGNOSTICS_DESCRIPTION)        
        print(treatment_diagnostics_df_t)
      
        if len(no_control_conditions) > 0:
            if len(no_control_conditions) == 1:
                print(f"NOTE: Treatment {no_control_conditions[0]} has no control conditions.")
            else:
                print(f"NOTE: Treatments {', '.join(no_control_conditions)} have no control conditions.")  

        print("-" * total_width)        

        data_diagnostics_df = self.data_diagnostics()

        width = data_diagnostics_df[config.DATA_DIAGNOSTICS_DESCRIPTION].str.len().max()
        data_diagnostics_df[config.DATA_DIAGNOSTICS_DESCRIPTION] = data_diagnostics_df[config.DATA_DIAGNOSTICS_DESCRIPTION].str.ljust(width)

        print(config.DATA_DIAGNOSTICS_DESCRIPTION)
        print(data_diagnostics_df.to_string(index=False, header=False))

        print("-" * total_width)        

        model_fit_metrics = self.fit_metrics()

        width = model_fit_metrics[config.MODEL_FIT_METRICS_DESCRIPTION].str.len().max()
        model_fit_metrics[config.MODEL_FIT_METRICS_DESCRIPTION] = model_fit_metrics[config.MODEL_FIT_METRICS_DESCRIPTION].str.ljust(width)
        model_fit_metrics[config.DIAGNOSTICS_COLUMN] = model_fit_metrics[config.DIAGNOSTICS_COLUMN].round(config.ROUND_STATISTIC)
        
        print(f"{config.MODEL_FIT_METRICS_DESCRIPTION}s")
        print(model_fit_metrics.to_string(index=False, header=False))
        
        print("=" * total_width)

        return self

    def plot_treatment_effects(
        self,
        colors = ["blue", "grey"],
        colors_by_signficance = ["red", "coral", "dimgray", "silver", "green", "palegreen"],
        point_type = "s",
        point_size = 8,
        line_width = 6,
        line_cap_size = 5, 
        x_label = "Estimates with confidence intervals",
        y_label = "Coefficient",
        plot_title = "DiD effects",
        plot_grid: bool = True,
        sort_by_coef: bool = False,
        sort_ascending: bool = True,
        plot_size: list = [7, 6],
        scale_plot: bool = True,
        show_central_tendency: bool = False,
        central_tendency: str = "mean"
        ):
           
        """
        Plot treatment effect estimates with confidence intervals.

        Parameters
        ----------
        colors : list, optional
            Color palette for points/bars. Default is ['blue', 'grey'].
        colors_by_signficance : list, optional
            Colors used to indicate significance thresholds. Default is ['red','coral','dimgray','silver','green','palegreen'].
        point_type : str, optional
            Marker type for point estimates. Default is 's'.
        point_size : int, optional
            Marker size for point estimates. Default is 8.
        line_width : int, optional
            Line width for error bars. Default is 6.
        line_cap_size : int, optional
            Cap size for error bars. Default is 5.
        x_label : str, optional
            Label for the x-axis. Default is 'Estimates with confidence intervals'.
        y_label : str, optional
            Label for the y-axis. Default is 'Coefficient'.
        plot_title : str, optional
            Plot title. Default is 'DiD effects'.
        plot_grid : bool, optional
            Show grid on plot. Default is True.
        sort_by_coef : bool, optional
            Sort effects by coefficient. Default is False.
        sort_ascending : bool, optional
            Sort order when sorting by coefficient. Default is True.
        plot_size : list, optional
            Figure size as [width, height]. Default is [7, 6].
        scale_plot : bool, optional
            Scale x-axis to fit estimates and intervals. Default is True.
        show_central_tendency : bool, optional
            Show mean/median line for estimates. Default is False.
        central_tendency : str, optional
            'mean' or 'median' for central tendency if shown. Default is 'mean'.

        Returns
        -------
        DiffModel
            Returns self to allow method chaining.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model.plot_treatment_effects()
        """

        model_config = self.data[1]

        confint_alpha = model_config["confint_alpha"]    
        
        treatment_effects = self.treatment_effects()

        if sort_by_coef:
            treatment_effects = treatment_effects.sort_values(
                by = treatment_effects.columns[1],
                ascending = not sort_ascending
                )
        
        plt.figure(figsize=(plot_size[0], plot_size[1]))
        
        point_estimates = treatment_effects["Estimate"].values
        CI_lower = treatment_effects["CI lower"].values
        CI_upper = treatment_effects["CI upper"].values
        treatment_coefs = treatment_effects.iloc[:, 0]

        if colors_by_signficance is None or colors_by_signficance == []:
            colors_by_signficance = [colors[0], colors[1], colors[0], colors[1], colors[0], colors[1]]

        for i, row in treatment_effects.iterrows():
            
            if row['p'] < confint_alpha and row['Estimate'] < 0:

                point_color = colors_by_signficance[0] 
                bar_color = colors_by_signficance[1]
            
            elif row['p'] < confint_alpha and row['Estimate'] > 0:

                point_color = colors_by_signficance[4]
                bar_color = colors_by_signficance[5]
            
            else:
            
                point_color = colors_by_signficance[2] 
                bar_color = colors_by_signficance[3]
            
            plt.errorbar(
                x = point_estimates[i],
                y = treatment_coefs[i],
                xerr = [[point_estimates[i] - CI_lower[i]], [CI_upper[i] - point_estimates[i]]],
                fmt = point_type,
                color = point_color,
                ecolor = bar_color,
                elinewidth = line_width,
                capsize = line_cap_size,
                markersize = point_size
                )
            
        if show_central_tendency:
            if central_tendency == "median":
                ITE_ct = np.median(treatment_effects["Estimate"])
            else:
                ITE_ct = np.mean(treatment_effects["Estimate"])
            plt.axvline(x = ITE_ct, color = "black")
        else:
            pass

        if scale_plot:
            maxval = treatment_effects.iloc[:, [1, 5, 6]].abs().max().max()
            maxval_plot = maxval*1.1
            plt.xlim(-maxval_plot, maxval_plot)
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(plot_title, fontsize=14)
        if plot_grid:
            plt.grid(True)
        
        plt.show()

        return self

    def is_parallel(self):

        """
        Proxy to tools.is_parallel using this model's data and config.

        Returns
        -------
        statsmodels regression or None
            The model used for the parallel trends test, if available.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model_is_parallel=model.is_parallel()
        >>> print(model_is_parallel)
        """

        model_data = self.data[2]
        model_config = self.data[1]

        modeldata_isparallel = tools.is_parallel(
            data = model_data,
            unit_col = model_config["unit_col"],
            time_col = model_config["time_col"],
            treatment_col = model_config["treatment_col"],
            outcome_col = model_config["outcome_col"],
            pre_post = model_config["pre_post"]
            )
        
        if modeldata_isparallel is not None:
            return modeldata_isparallel[1]
        else:
            return None
    
    def predictions(self):
        
        """
        Return model predictions (summary frame or array-like).

        Returns
        -------
        pandas.DataFrame or array-like: Model predictions summary frame or array-like.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model_predictions=model.predictions()
        >>> print(model_predictions)
        """

        model_predictions = self.data[3]

        return model_predictions
    
    def counterfactual(
        self,
        treatment = None,
        after_treatment_col: str = None
        ):
        
        """
        Compute counterfactual predictions for the treatment group by setting treatment to zero.

        Parameters
        ----------
        treatment : str, optional
            Treatment name to analyse. If None, the first treatment is chosen.
        after_treatment_col : str, optional
            After-treatment indicator column name.

        Returns
        -------
        list
            [modified_model_data (DataFrame), outcome_pred_col (str), outcome_pred_cf_col (str)]

        Raises
        ------
        ValueError
            If requested treatment is not present in the model object.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model_counterfactual=model.counterfactual()
        >>> print(model_counterfactual[0])
        """

        model_config = self.data[1]
        outcome_col = model_config["outcome_col"]

        model_data = self.data[2]
             
        if treatment is not None:
            if treatment not in model_config["treatment_col"]:
                raise ValueError (f"Treatment {treatment} not in model object.")
        else:
            treatment = model_config["treatment_col"][0]
            print(f"NOTE: No treatment was stated. Choosing treatment {treatment} for analysis.")

        if after_treatment_col is not None:
            if len(model_config["after_treatment_col"]) == 0:
                raise ValueError ("Model object does not include after-treatment period.")
            if after_treatment_col not in model_config["after_treatment_col"]:
                raise ValueError (f"After-treatment variable {after_treatment_col} not in model object.")
        else:
            after_treatment_col = []

        didmodel = self.didmodel()

        predictions = self.predictions()
        
        model_data = self.data[2]
        
        model_config = self.data[1]
 
        model_data_mod = model_data.copy()
        model_data_mod[treatment] = 0
        if after_treatment_col is not None:
            model_data_mod[after_treatment_col] = 0

        predictions_counterfac = didmodel.get_prediction(model_data_mod).summary_frame()

        outcome_pred_col = f"{outcome_col}{config.PREDICTED_SUFFIX}"
        if outcome_pred_col in model_data_mod.columns:
            outcome_pred_col = f"{config.DELIMITER}{outcome_pred_col}"
        outcome_pred_cf_col = f"{outcome_col}{config.COUNTERFAC_SUFFIX_PRED_CF}"
        if outcome_pred_cf_col in model_data_mod.columns:
            outcome_pred_cf_col = f"{config.DELIMITER}{outcome_pred_cf_col}"

        model_data_mod[outcome_pred_col] = predictions[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[0]]
        model_data_mod[outcome_pred_cf_col] = predictions_counterfac[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[0]]

        return [
            model_data_mod,
            outcome_pred_col,
            outcome_pred_cf_col
        ]

    def didmodel(self):
        
        """
        Return the fitted did model object (estimator/result).

        Returns
        -------
        object: Underlying model object used for prediction.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> did_model=model.didmodel()
        """

        did_model = self.data[4]
        return did_model
    
    def prediction_intervals(
        self,
        confint_alpha = 0.05
        ):

        """
        Return prediction intervals from the fitted model.

        Parameters
        ----------
        confint_alpha : float, optional
            Significance level for the intervals.

        Returns
        -------
        pandas.DataFrame
            Summary frame with prediction and interval columns.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model_prediction_intervals=model.prediction_intervals()
        >>> print(model_prediction_intervals)
        """

        did_model = self.data[4]

        prediction_intervals = did_model.get_prediction()
        prediction_intervals = prediction_intervals.summary_frame(alpha = confint_alpha)

        return prediction_intervals
    
    def placebo(
        self,
        treatment: str = None,
        after_treatment_col: str = None,
        TG_col: str = None,
        TT_col: str = None,
        divide: float = 0.5,
        resample: float = 1.0,
        random_state = 71
        ):

        """
        Run a placebo test by resampling control units and estimating a placebo DiD.

        Parameters
        ----------
        treatment : str, optional
            Treatment name to use for placebo assignment.
        TG_col : str, optional
            Column name identifying treatment group.
        TT_col : str, optional
            Column name identifying treatment time.
        divide : float, optional
            Share of control units to flip into placebo-treated (0<divide<=1).
        resample : float, optional
            Fraction of the divided group to sample when creating placebo units.
        random_state : int, optional
            Seed for random sampling.

        Returns
        -------
        DiffModel
            Resulting placebo analysis object.

        Raises
        ------
        ValueError
            If parameters are invalid or required TG/TT columns are missing.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> placebo_analysis=model.placebo()
        >>> placebo_analysis.summary()
        """

        model_config = self.data[1]
        model_data = self.data[2]
             
        if treatment is not None:
            if treatment not in model_config["treatment_col"]:
                raise ValueError (f"Treatment {treatment} not in model object.")
        else:
            treatment = model_config["treatment_col"][0]
            print(f"NOTE: No treatment was stated. Choosing treatment {treatment} for analysis.")

        if after_treatment_col is not None:
            if len(model_config["after_treatment_col"]) == 0:
                raise ValueError ("Model object does not include after-treatment period.")
            if after_treatment_col not in model_config["after_treatment_col"]:
                raise ValueError (f"Treatment {treatment} not in model object.")
        else:
            after_treatment_col = []

        if divide <= 0 or divide > 1:
            raise ValueError("Parameter share must be > 0 and <= 1")
        if resample <= 0 or resample > 1:
            raise ValueError("Parameter resample must be > 0 and <= 1")
        
        treatment_statistics = self.treatment_statistics(treatment = treatment)
        
        TG_col_ = f"{config.TG_COL}{config.DELIMITER}{treatment}"
        TT_col_ = f"{config.TT_COL}{config.DELIMITER}{treatment}"
        TGxTT_ = f"Placebo{config.DELIMITER}{treatment}"
        
        if TG_col is None and TG_col_ not in model_config["TG_col"]:
            raise ValueError(f"No treatment group identification variable for treatment {treatment}. Please state TG_col = your_treatment_group_dummy.")
        
        if TT_col is None and TT_col_ not in model_config["TT_col"]:
            raise ValueError(f"No treatment time variable for treatment {treatment}. Please state TG_col = your_treatment_time_dummy.")
        
        if TG_col is not None:
            TG_col_ = TG_col
        if TT_col is not None:
            TT_col_ = TT_col

        unit_col = model_config["unit_col"]
        time_col = model_config["time_col"]
        
        groups = treatment_statistics[2]
        control_group = groups[1]
        control_group_N = len(control_group)

        time_periods = treatment_statistics[4]
        treatment_period_start = time_periods[1][0]
        treatment_period_end = time_periods[1][1]
        treatment_period_start = pd.to_datetime(treatment_period_start)
        treatment_period_end = pd.to_datetime(treatment_period_end)

        model_data_c = model_data[model_data[unit_col].isin(control_group)].copy()
        model_data_c[time_col] = pd.to_datetime(model_data_c[time_col])
        model_data_c[unit_col] = model_data_c[unit_col].astype(str)

        units_random_sample = model_data_c[unit_col].sample(
            n = int(round(divide*control_group_N*resample, 0)), 
            random_state = random_state
            ).astype(str).tolist()

        model_data_c[TG_col_] = 0
        model_data_c.loc[(model_data_c[unit_col].isin(units_random_sample)), TG_col_] = 1
        model_data_c[TGxTT_] = model_data_c[TG_col_] * model_data_c[TT_col_]

        model_data_c_analysis = did_analysis(
            data = model_data_c,
            unit_col = unit_col,
            time_col = time_col,
            treatment_col = TGxTT_,
            outcome_col = model_config["outcome_col"],
            TG_col = TG_col_,
            TT_col = TT_col_,
            after_treatment_col = after_treatment_col,
            pre_post = model_config["pre_post"],
            log_outcome = model_config["log_outcome"],
            FE_unit = model_config["FE_unit"],
            FE_time = model_config["FE_time"],
            ITE = model_config["ITE"],
            GTE = model_config["GTE"],
            ITT = model_config["ITT"],
            GTT = model_config["GTT"],
            group_by = model_config["group_by"],
            covariates = model_config["covariates"], 
            confint_alpha = model_config["confint_alpha"],
            drop_missing = model_config["drop_missing"],
            placebo = True    
            )
            
        return model_data_c_analysis

    def plot_timeline(
        self,
        treatment: str = None,
        TG_col: str = None,
        x_label = "Time",
        y_label = "Analysis units",
        y_lim = None,
        plot_title = "Treatment time",
        plot_symbol = "o",
        treatment_group_only = True
        ):

        """
        Plot treatment timing across units (timeline of intervention).

        Parameters
        ----------
        treatment : str, optional
            Treatment name to plot.
        TG_col : str, optional
            Column identifying treatment group.
        x_label : str, optional
            Label for the x-axis. Default is 'Time'.
        y_label : str, optional
            Label for the y-axis. Default is 'Analysis units'.
        y_lim : tuple or list, optional
            y-axis limits as (ymin, ymax). Default is None.
        plot_title : str, optional
            Plot title. Default is 'Treatment time'.
        plot_symbol : str, optional
            Symbol used for treatment timing points. Default is 'o'.
        treatment_group_only : bool, optional
            If True, only plot treated units.

        Returns
        -------
        pandas.DataFrame
            Pivot table used for plotting (time x units).

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model.plot_timeline(treatment='Curfew')
        """

        model_config = self.data[1]
        model_data = self.data[2]

        if treatment is not None:
            if treatment not in model_config["treatment_col"]:
                raise ValueError (f"Treatment {treatment} not in model object.")
        else:
            treatment = model_config["treatment_col"][0]
            print(f"NOTE: No treatment was stated. Choosing treatment {treatment} for analysis.")
                
        if treatment_group_only:
            if TG_col is None:
                raise ValueError("Set TG_col = [treament_group_col] to identify treatment group.") 

        modeldata_pivot = model_data.pivot_table (
            index = model_config["time_col"],
            columns = model_config["unit_col"],
            values = treatment
            )

        fig, ax = plt.subplots(figsize=(12, len(modeldata_pivot.columns) * 0.5))

        modeldata_pivot.index = pd.to_datetime(modeldata_pivot.index)

        for i, col in enumerate(modeldata_pivot.columns):
            time_points_treatment = modeldata_pivot.index[modeldata_pivot[col] == 1]
            values = [i] * len(time_points_treatment)
            ax.plot(time_points_treatment, values, plot_symbol, label=col)

        ax.set_xlabel(x_label)
        ax.set_yticks(range(len(modeldata_pivot.columns)))
        ax.set_yticklabels(modeldata_pivot.columns)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        ax.xaxis.set_major_formatter(DateFormatter(model_config["date_format"]))
        
        plt.xticks(rotation=90)
        plt.tight_layout()

        start_date = min(modeldata_pivot.index)
        end_date = max(modeldata_pivot.index)
        ax.set_xlim(start_date, end_date)
        
        if y_lim is not None:
            ax.set_ylim(y_lim)

        plt.show()

        return modeldata_pivot

    def plot(
        self,
        treatment = None,
        x_label: str = "Time",
        y_label: str = "Outcome",
        y_lim = None,
        plot_title: str = "Treatment group vs. control group",
        lines_col: list = ["blue", "green", "red", "orange"],
        lines_style: list = ["solid", "solid", "dashed", "dashed"],
        lines_labels: list = ["TG observed", "CG observed", "TG fit", "CG fit", "TG fit CI", "CG fit CI"],
        plot_legend: bool = True,
        plot_grid: bool = True,
        plot_observed: bool = False,
        plot_intervals: str = "confint",
        plot_intervals_groups: list = ["TG", "CG"],
        plot_size_auto: bool = True,
        plot_size: list = [12, 6],
        pre_post_ticks: list = ["Pre", "Post"],
        pre_post_barplot = False,
        pre_post_bar_width = 0.5      
        ):

        """
        Plot observed and expected outcome series for treatment and control groups.

        Parameters
        ----------
        treatment : str, optional
            Which treatment to plot (if multiple are present).
        x_label : str, optional
            Label for the x-axis. Default is 'Time'.
        y_label : str, optional
            Label for the y-axis. Default is 'Outcome'.
        y_lim : tuple or list, optional
            y-axis limits as (ymin, ymax). Default is None.
        plot_title : str, optional
            Plot title. Default is 'Treatment group vs. control group'.
        lines_col : list, optional
            Colors for plotted lines. Default is ['blue','green','red','orange'].
        lines_style : list, optional
            Line styles for plotted lines. Default is ['solid','solid','dashed','dashed'].
        lines_labels : list, optional
            Labels for plotted series. Default includes observed and fit labels.
        plot_legend : bool, optional
            Show legend. Default is True.
        plot_grid : bool, optional
            Show grid. Default is True.
        plot_observed : bool, optional
            Plot observed series in addition to fits. Default is False.
        plot_intervals : str, optional
            'confint' or 'predict' to plot intervals. Default is 'confint'.
        plot_intervals_groups : list, optional
            Which groups to include intervals for, e.g., ['TG','CG']. Default is ['TG','CG'].
        plot_size_auto : bool, optional
            Let function pick figure size automatically. Default is True.
        plot_size : list, optional
            If not auto, figure size [width, height]. Default is [12,6].
        pre_post_ticks : list, optional
            Tick labels for pre/post barplot. Default is ['Pre','Post'].
        pre_post_barplot : bool, optional
            Plot pre-post as barplot. Default is False.
        pre_post_bar_width : float, optional
            Bar width when pre_post_barplot is True. Default is 0.5.

        Returns
        -------
        pandas.DataFrame
            Aggregated data used for plotting (means by time and group).

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     treatment_name="Curfew",
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model.plot(treatment='Curfew')
        """

        model_config = self.data[1]
        TG_col = model_config["TG_col"]
        unit_col = model_config["unit_col"]
        treatment_diagnostics = model_config["treatment_diagnostics"]
        no_treatments = model_config["no_treatments"]
        outcome_col = model_config["outcome_col"]
        outcome_col_predicted = f"{outcome_col}{config.PREDICTED_SUFFIX}"

        if TG_col is None and treatment is None:            
            if no_treatments == 1:
                raise ValueError ("Model object has no column for treatment group with respect to one treatment. Set parameter treatment = [your_treatment].")
            else:
                raise ValueError ("Model object has no column for treatment group with respect to ", str(no_treatments), " treatments. Choose one with parameter treatment.")

        if treatment is not None:
            
            treatment_included = any(
                entry.get("treatment") == treatment
                for entry in treatment_diagnostics.values()
                )
                  
            if not treatment_included:                
                raise ValueError (f"Treatment {treatment} not in model object")
            
            for key, value in treatment_diagnostics.items():
                if value["treatment"] == treatment:                
                    treatment_group = value["treatment_group"]
                    break                        
        else:
            
            print(f"NOTE: No treatment was stated. Choosing treatment {treatment_diagnostics[0]['treatment']} for plotting.")

            treatment_group = treatment_diagnostics[0]["treatment_group"]            
            treatment = treatment_diagnostics[0]["treatment"]

        lines_col_required = 4
        lines_style_required = lines_col_required
        lines_labels_required = lines_col_required
        if ("TG" in plot_intervals_groups and "CG" not in plot_intervals_groups) or ("CG" in plot_intervals_groups and "TG" not in plot_intervals_groups):
            lines_labels_required = lines_labels_required+1
        if "TG" in plot_intervals_groups and "CG" in plot_intervals_groups:
            lines_labels_required = lines_labels_required+2            
        assert len(lines_col) == lines_col_required, f"Parameter 'lines_col' must be a list with {lines_col_required} entries"
        assert len(lines_style) == lines_style_required, f"Parameter 'lines_style' must be a list with {lines_style_required} entries"
        assert len(lines_labels) == lines_labels_required, f"Parameter 'lines_labels' must be a list with {lines_labels_required} entries"
            
        model_data = self.data[2]
        model_data = model_data.reset_index()
        TG_col = f"{config.TG_COL}{config.DELIMITER}{treatment}"
        model_data[TG_col] = 0
        model_data.loc[model_data[unit_col].isin(treatment_group), TG_col] = 1

        model_predictions = self.data[3]
        model_predictions = pd.DataFrame(model_predictions)
        model_predictions = model_predictions.reset_index()
        model_predictions.rename(columns = {config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[0]: outcome_col_predicted}, inplace = True)
    
        model_data = pd.concat ([model_data, model_predictions], axis = 1)
        
        model_data_TG = model_data[model_data[TG_col] == 1]
        model_data_CG = model_data[model_data[TG_col] == 0]
    
        model_data_TG_mean = model_data_TG.groupby(model_config["time_col"])[outcome_col].mean()
        model_data_TG_mean = model_data_TG_mean.reset_index()
        model_data_CG_mean = model_data_CG.groupby(model_config["time_col"])[outcome_col].mean()
        model_data_CG_mean = model_data_CG_mean.reset_index()
    
        model_data_TG_mean_pred = model_data_TG.groupby(model_config["time_col"])[outcome_col_predicted].mean()
        model_data_TG_mean_pred = model_data_TG_mean_pred.reset_index()
        model_data_CG_mean_pred = model_data_CG.groupby(model_config["time_col"])[outcome_col_predicted].mean()
        model_data_CG_mean_pred = model_data_CG_mean_pred.reset_index()
        
        model_data_TG_CI_lower_mean_pred = model_data_TG.groupby(model_config["time_col"])[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[2]].mean()
        model_data_TG_CI_lower_mean_pred = model_data_TG_CI_lower_mean_pred.reset_index()        
        model_data_CG_CI_lower_mean_pred = model_data_CG.groupby(model_config["time_col"])[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[2]].mean()
        model_data_CG_CI_lower_mean_pred = model_data_CG_CI_lower_mean_pred.reset_index()
        
        model_data_TG_CI_upper_mean_pred = model_data_TG.groupby(model_config["time_col"])[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[3]].mean()
        model_data_TG_CI_upper_mean_pred = model_data_TG_CI_upper_mean_pred.reset_index()        
        model_data_CG_CI_upper_mean_pred = model_data_CG.groupby(model_config["time_col"])[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[3]].mean()
        model_data_CG_CI_upper_mean_pred = model_data_CG_CI_upper_mean_pred.reset_index()

        model_data_TG_PI_lower_mean_pred = model_data_TG.groupby(model_config["time_col"])[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[4]].mean()
        model_data_TG_PI_lower_mean_pred = model_data_TG_PI_lower_mean_pred.reset_index()        
        model_data_CG_PI_lower_mean_pred = model_data_CG.groupby(model_config["time_col"])[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[4]].mean()
        model_data_CG_PI_lower_mean_pred = model_data_CG_PI_lower_mean_pred.reset_index()
        
        model_data_TG_PI_upper_mean_pred = model_data_TG.groupby(model_config["time_col"])[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[5]].mean()
        model_data_TG_PI_upper_mean_pred = model_data_TG_PI_upper_mean_pred.reset_index()        
        model_data_CG_PI_upper_mean_pred = model_data_CG.groupby(model_config["time_col"])[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[5]].mean()
        model_data_CG_PI_upper_mean_pred = model_data_CG_PI_upper_mean_pred.reset_index()
    
        model_data_TG_CG = pd.concat ([
            model_data_TG_mean.reset_index(drop=True),
            model_data_CG_mean[outcome_col].reset_index(drop=True),
            model_data_TG_mean_pred[outcome_col_predicted].reset_index(drop=True),
            model_data_CG_mean_pred[outcome_col_predicted].reset_index(drop=True),
            model_data_TG_CI_lower_mean_pred[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[2]].reset_index(drop=True),
            model_data_TG_CI_upper_mean_pred[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[3]].reset_index(drop=True),
            model_data_CG_CI_lower_mean_pred[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[2]].reset_index(drop=True),
            model_data_CG_CI_upper_mean_pred[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[3]].reset_index(drop=True),
            model_data_TG_PI_lower_mean_pred[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[4]].reset_index(drop=True),
            model_data_TG_PI_upper_mean_pred[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[5]].reset_index(drop=True),
            model_data_CG_PI_lower_mean_pred[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[4]].reset_index(drop=True),
            model_data_CG_PI_upper_mean_pred[config.PREDICTIONS_SUMMARY_FRAME_COLS_LIST[5]].reset_index(drop=True),
            ],
            axis = 1)
        
        outcome_col_observed_TG = f"{outcome_col}{config.DELIMITER}{config.OBSERVED_SUFFIX}{config.DELIMITER}{config.TG_COL}"
        outcome_col_observed_CG = f"{outcome_col}{config.DELIMITER}{config.OBSERVED_SUFFIX}{config.DELIMITER}{config.CG_COL}"
        outcome_col_expected_TG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.TG_COL}"
        outcome_col_expected_CG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.CG_COL}"
        outcome_col_expected_CI_lower_TG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.CI_LOWER_SUFFIX}{config.DELIMITER}{config.TG_COL}"
        outcome_col_expected_CI_upper_TG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.CI_UPPER_SUFFIX}{config.DELIMITER}{config.TG_COL}"
        outcome_col_expected_CI_lower_CG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.CI_LOWER_SUFFIX}{config.DELIMITER}{config.CG_COL}"
        outcome_col_expected_CI_upper_CG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.CI_UPPER_SUFFIX}{config.DELIMITER}{config.CG_COL}"

        outcome_col_expected_PI_lower_TG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.PI_LOWER_SUFFIX}{config.DELIMITER}{config.TG_COL}"
        outcome_col_expected_PI_upper_TG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.PI_UPPER_SUFFIX}{config.DELIMITER}{config.TG_COL}"
        outcome_col_expected_PI_lower_CG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.PI_LOWER_SUFFIX}{config.DELIMITER}{config.CG_COL}"
        outcome_col_expected_PI_upper_CG = f"{outcome_col}{config.DELIMITER}{config.EXPECTED_SUFFIX}{config.DELIMITER}{config.PI_UPPER_SUFFIX}{config.DELIMITER}{config.CG_COL}"

        model_data_TG_CG.columns.values[0] = "t"
        model_data_TG_CG.columns.values[1] = outcome_col_observed_TG
        model_data_TG_CG.columns.values[2] = outcome_col_observed_CG
        model_data_TG_CG.columns.values[3] = outcome_col_expected_TG
        model_data_TG_CG.columns.values[4] = outcome_col_expected_CG
        model_data_TG_CG.columns.values[5] = outcome_col_expected_CI_lower_TG
        model_data_TG_CG.columns.values[6] = outcome_col_expected_CI_upper_TG
        model_data_TG_CG.columns.values[7] = outcome_col_expected_CI_lower_CG
        model_data_TG_CG.columns.values[8] = outcome_col_expected_CI_upper_CG
        model_data_TG_CG.columns.values[9] = outcome_col_expected_PI_lower_TG
        model_data_TG_CG.columns.values[10] = outcome_col_expected_PI_upper_TG
        model_data_TG_CG.columns.values[11] = outcome_col_expected_PI_lower_CG
        model_data_TG_CG.columns.values[12] = outcome_col_expected_PI_upper_CG
               
        if plot_size_auto:
            if model_config["pre_post"]:
                fig, ax = plt.subplots(figsize=(7, 6))
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig, ax = plt.subplots(figsize=(plot_size[0], plot_size[1]))       
    
        model_data_TG_CG["t"] = pd.to_datetime(model_data_TG_CG["t"])

        if not model_config["pre_post"]:
            pre_post_barplot = False        

        if pre_post_barplot:

            x_pos_t1_TG = 0
            x_pos_t1_CG = x_pos_t1_TG + pre_post_bar_width  
            x_pos_t2_TG = 1.5  
            x_pos_t2_CG = x_pos_t2_TG + pre_post_bar_width  

            plt.bar(
                x = x_pos_t1_TG, 
                height = model_data_TG_CG[outcome_col_expected_TG][0], 
                label = lines_labels[2], 
                color = lines_col[2], 
                width = pre_post_bar_width
                )   
            plt.bar(
                x = x_pos_t1_CG, 
                height = model_data_TG_CG[outcome_col_expected_CG][0], 
                label = lines_labels[3], 
                color = lines_col[3], 
                width = pre_post_bar_width
                )            
            plt.bar(
                x = x_pos_t2_TG, 
                height = model_data_TG_CG[outcome_col_expected_TG][1],                 
                color = lines_col[2], 
                width = pre_post_bar_width
                )            
            plt.bar(
                x = x_pos_t2_CG, 
                height = model_data_TG_CG[outcome_col_expected_CG][1],                 
                color=lines_col[3], 
                width = pre_post_bar_width
                )

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(plot_title)
            
        else:

            if plot_observed:
                
                plt.plot(
                    model_data_TG_CG["t"], 
                    model_data_TG_CG[outcome_col_observed_TG], 
                    label = lines_labels[0], 
                    color=lines_col[0], 
                    linestyle=lines_style[0]
                    )
                plt.plot(
                    model_data_TG_CG["t"], 
                    model_data_TG_CG[outcome_col_observed_CG], 
                    label = lines_labels[1], 
                    color=lines_col[1], 
                    linestyle=lines_style[1]
                    )
            
            plt.plot(
                model_data_TG_CG["t"], 
                model_data_TG_CG[outcome_col_expected_TG],
                label=lines_labels[2],
                color=lines_col[2], 
                linestyle=lines_style[2]
                )
            plt.plot(
                model_data_TG_CG["t"], 
                model_data_TG_CG[outcome_col_expected_CG], 
                label=lines_labels[3], 
                color=lines_col[3], 
                linestyle=lines_style[3]
                )

            if plot_intervals == "confint":
                
                if "TG" in plot_intervals_groups:
                
                    plt.fill_between(
                        model_data_TG_CG["t"],
                        model_data_TG_CG[outcome_col_expected_CI_lower_TG],
                        model_data_TG_CG[outcome_col_expected_CI_upper_TG],
                        label=lines_labels[4],
                        color=lines_col[2],
                        alpha=0.2
                    )
                    
                if "CG" in plot_intervals_groups:
                
                    plt.fill_between(
                        model_data_TG_CG["t"],
                        model_data_TG_CG[outcome_col_expected_CI_lower_CG],
                        model_data_TG_CG[outcome_col_expected_CI_upper_CG],
                        label=lines_labels[5],
                        color=lines_col[3],
                        alpha=0.2
                    )

            if plot_intervals == "predict":

                if "TG" in plot_intervals_groups:
                
                    plt.fill_between(
                        model_data_TG_CG["t"],
                        model_data_TG_CG[outcome_col_expected_PI_lower_TG],
                        model_data_TG_CG[outcome_col_expected_PI_upper_TG],
                        label=lines_labels[4],
                        color=lines_col[2],
                        alpha=0.2
                    )
                    
                if "CG" in plot_intervals_groups:
                
                    plt.fill_between(
                        model_data_TG_CG["t"],
                        model_data_TG_CG[outcome_col_expected_PI_lower_CG],
                        model_data_TG_CG[outcome_col_expected_PI_upper_CG],
                        label=lines_labels[5],
                        color=lines_col[3],
                        alpha=0.2
                    )
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(plot_title)
            ax.xaxis.set_major_formatter(DateFormatter(model_config["date_format"]))

        if model_config["pre_post"]:
            if not pre_post_barplot:
                plt.xticks(
                    model_data_TG_CG["t"].unique(), 
                    labels = [pre_post_ticks[0], pre_post_ticks[1]]
                    )  
            else:
                plt.xticks(
                    [0.25, 1.75], 
                    labels = [pre_post_ticks[0], pre_post_ticks[1]]
                    )  
        else:
            plt.xticks(rotation=90)
        
        plt.tight_layout()

        if plot_legend:
            plt.legend()

        if plot_grid:
            if not model_config["pre_post"]:
                plt.grid(True)
            else:
                plt.grid(
                    axis='y', 
                    linestyle='-', 
                    alpha=0.7
                    )

        if y_lim is not None:
            ax.set_ylim(y_lim)
            
        plt.show()
        
        return model_data_TG_CG    

    def plot_counterfactual(
        self,
        treatment: str = None,
        after_treatment_col: str = None,
        x_label: str = "Time",
        y_label: str = "Outcome",
        y_lim = None,
        plot_title: str = "Treatment group Counterfactual",
        lines_col: list = ["blue", "green"],
        lines_style: list = ["solid", "dashed"],
        lines_labels: list = ["TG", "TG counterfactual"],
        plot_legend: bool = True,
        plot_grid: bool = True,
        plot_size: list = [12, 6]
        ):

        """
        Plot treatment-group mean prediction and counterfactual over time.

        Parameters
        ----------
        treatment : str, optional
            Treatment name to plot.
        after_treatment_col : str, optional
            After-treatment column name.
        x_label : str, optional
            Label for the x-axis. Default is 'Time'.
        y_label : str, optional
            Label for the y-axis. Default is 'Outcome'.
        y_lim : tuple or list, optional
            y-axis limits as (ymin, ymax). Default is None.
        plot_title : str, optional
            Plot title. Default is 'Treatment group Counterfactual'.
        lines_col : list, optional
            Colors for the plotted lines. Default is ['blue','green'].
        lines_style : list, optional
            Line styles for plotted lines. Default is ['solid','dashed'].
        lines_labels : list, optional
            Labels for plotted series. Default is ['TG','TG counterfactual'].
        plot_legend : bool, optional
            Show legend. Default is True.
        plot_grid : bool, optional
            Show grid. Default is True.
        plot_size : list, optional
            Figure size as [width, height]. Default is [12, 6].

        Returns
        -------
        pandas.DataFrame
            Mean predicted vs. counterfactual for treated units.

        Examples
        --------
        >>> curfew_data=create_data(
        ...     outcome_data=curfew_DE,
        ...     unit_id_col='county',
        ...     time_col='infection_date',
        ...     outcome_col='infections_cum_per100000',
        ...     treatment_group=curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     control_group=curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        ...     study_period=['2020-03-01','2020-05-15'],
        ...     treatment_period=['2020-03-21','2020-05-05'],
        ...     treatment_name="Curfew",
        ...     freq='D'
        ... )
        >>> model=curfew_data.analysis()
        >>> model.plot_counterfactual(treatment='Curfew')
        """

        model_config = self.data[1]
        TG_col = model_config["TG_col"]
        time_col = model_config["time_col"]
        unit_col = model_config["unit_col"]
        no_treatments = model_config["no_treatments"]
        treatment_diagnostics = model_config["treatment_diagnostics"]

        if TG_col is None and treatment is None:
            
            if no_treatments == 1:
                raise ValueError ("Model object has no column for treatment group with respect to one treatment. Set parameter treatment = [your_treatment].")
            else:
                raise ValueError ("Model object has no column for treatment group with respect to ", str(no_treatments), " treatments. Choose one with parameter treatment.")

        counterfac_results = self.counterfactual(
            treatment = treatment,
            after_treatment_col = after_treatment_col
            )

        model_data_mod = counterfac_results[0]
        outcome_col_pred = counterfac_results[1]
        outcome_col_pred_counterfac = counterfac_results[2]

        if treatment is not None:

            treatment_included = any(
                entry.get("treatment") == treatment
                for entry in treatment_diagnostics.values()
                )
            
            if not treatment_included:
                raise ValueError (f"Treatment {treatment} not in model object")
            
            for key, value in treatment_diagnostics.items():
                if value["treatment"] == treatment:
                    treatment_group = value["treatment_group"]
                    break
    
        else:
            print(f"NOTE: No treatment was stated. Choosing treatment {treatment_diagnostics[0]['treatment']} for plotting.")

            treatment_group = treatment_diagnostics[0]["treatment_group"]

            treatment = treatment_diagnostics[0]["treatment"]

        treatment_group = [str(x) for x in treatment_group]

        TG_col = f"{config.TG_COL}{config.DELIMITER}{treatment}"
        
        model_data_mod[TG_col] = 0
        model_data_mod.loc[model_data_mod[unit_col].astype(str).isin(treatment_group), TG_col] = 1

        model_data_mod_TG = model_data_mod.loc[model_data_mod[TG_col] == 1]  

        model_data_TG_mean_pred = model_data_mod_TG.groupby(time_col)[outcome_col_pred].mean()
        model_data_TG_mean_pred = model_data_TG_mean_pred.reset_index()
        
        model_data_TG_mean_pred_counterfac = model_data_mod_TG.groupby(time_col)[outcome_col_pred_counterfac].mean()
        model_data_TG_mean_pred_counterfac = model_data_TG_mean_pred_counterfac.reset_index()
        model_data_TG_mean_pred_counterfac = model_data_TG_mean_pred_counterfac.drop(columns=[time_col])

        model_data_TG_mean = pd.concat ([
            model_data_TG_mean_pred.reset_index(),
            model_data_TG_mean_pred_counterfac.reset_index()
            ],
            axis = 1)
        model_data_TG_mean[time_col] = pd.to_datetime(model_data_TG_mean[time_col])

        fig, ax = plt.subplots(figsize=(plot_size[0], plot_size[1]))   
        
        plt.plot(
            model_data_TG_mean[time_col], 
            model_data_TG_mean[outcome_col_pred_counterfac], 
            label = lines_labels[1], 
            color = lines_col[1], 
            linestyle=lines_style[1]
            )
        plt.plot(
            model_data_TG_mean[time_col], 
            model_data_TG_mean[outcome_col_pred], 
            label = lines_labels[0], 
            color = lines_col[0], 
            linestyle = lines_style[0]
            )
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)
        ax.xaxis.set_major_formatter(DateFormatter(model_config["date_format"]))

        if plot_legend:
            plt.legend()

        if plot_grid:
            if not model_config["pre_post"]:
                plt.grid(True)
            else:
                plt.grid(axis='y', linestyle='-', alpha=0.7)
        
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if y_lim is not None:
            ax.set_ylim(y_lim)
            
        plt.show()

        return model_data_TG_mean

def did_analysis(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: list,
    outcome_col: str,
    TG_col: list = None,
    TT_col: list = None,
    after_treatment_col: list = None,
    ATT_col: list = None,
    pre_post: bool = False,
    log_outcome: bool = False,
    log_outcome_add = 0.01,
    FE_unit: bool = False,
    FE_time: bool = False,
    FE_group: bool = False,
    cluster_SE_by: str = None,
    intercept: bool = True,
    ITE: bool = False,
    GTE: bool = False,
    ITT: bool = False,
    GTT: bool = False,
    group_by: str = None,
    covariates: list = None,
    spillover_treatment: list = None,
    spillover_units: list = None,
    placebo: bool = False,
    confint_alpha = 0.05,
    bonferroni: bool = False,
    freq: str = "D",
    date_format: str = "%Y-%m-%d",
    drop_missing: bool = True,
    missing_replace_by_zero: bool = False,
    fit_by: str = "ols_fit",
    verbose: bool = config.VERBOSE
    ):    
   
    """
    Perform a Difference-in-Differences analysis with a given data frame.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data containing units, time and outcome.
    unit_col : str
        Name of unit identifier column.
    time_col : str
        Name of time column.
    treatment_col : list
        Treatment column(s) names.
    outcome_col : str
        Outcome variable name.
    TG_col : list
        Treatment group indicator column(s).
    TT_col : list
        Treatment time indicator column(s).
    after_treatment_col : list
        Columns indicating post-treatment periods.
    ATT_col : list
        Columns for average treatment on the treated calculations.
    pre_post : bool
        Treat data as pre-post panel.
    log_outcome : bool
        Log-transform the outcome.
    log_outcome_add : float
        Constant to add before log transformation.
    FE_unit : bool
        Include unit fixed effects.
    FE_time : bool
        Include time fixed effects.
    FE_group : bool
        Include group fixed effects.
    cluster_SE_by : str
        Column name to cluster standard errors by.
    intercept : bool
        Include intercept in the model.
    ITE : bool
        Estimate individual treatment effects.
    GTE : bool
        Estimate group treatment effects.
    ITT : bool
        Include individual time trends.
    GTT : bool
        Include group-specific time trends.
    group_by : str
        Column name defining groups.
    covariates : list
        Additional covariate columns to include.
    spillover_treatment : list
        Treatment columns used to construct spillover variables.
    spillover_units : list
        Unit identifiers affected by spillovers.
    placebo : bool
        Run placebo analysis.
    confint_alpha : float
        Significance level for confidence intervals.
    bonferroni : bool
        Apply Bonferroni correction for multiple treatments.
    freq : str
        Frequency string for date handling.
    date_format : str
        Date format string.
    drop_missing : bool
        Drop missing observations before analysis.
    missing_replace_by_zero : bool
        Replace missing values by zero when requested.
    fit_by : str
        Fitting method; e.g., 'ols_fit' or 'ml'.
    verbose : bool, optional
            If True, print progress messages.

    Returns
    -------
    DiffModel
        Container object with model results, config, data and helpers.

    Raises
    ------
    ValueError
        If required arguments are missing or invalid.

    Examples
    --------
    >>> Corona_Hesse=pd.read_excel("data/Corona_Hesse.xlsx")
    >>> Hesse_model1=did_analysis(
    ...     data=Corona_Hesse,
    ...     unit_col="REG_NAME",
    ...     time_col="infection_date",
    ...     treatment_col="Nighttime_curfew",
    ...     outcome_col="R7_rm",
    ...     intercept=False
    ...     )
    >>> Hesse_model1.summary()
    """

    if TG_col is None:
        TG_col = []
    if TT_col is None:
        TT_col = []
    if after_treatment_col is None:
        after_treatment_col = []
    if ATT_col is None:
        ATT_col = []
    if covariates is None:        
        covariates = []
    if spillover_treatment is None:
        spillover_treatment = []
    if spillover_units is None:
        spillover_units = []

    tools.check_columns(
        df = data,
        columns = [
            unit_col, 
            time_col, 
            outcome_col
            ],
        verbose = verbose
        )
    
    if isinstance (treatment_col, str):
        if treatment_col == "":
            raise ValueError("No treatment(s) in parameter 'treatment_col' stated.")
        treatment_col = [treatment_col]

    no_treatments = len(treatment_col)
    
    if no_treatments == 0:
        raise ValueError("No treatment(s) in parameter 'treatment_col' stated.")
    
    tools.check_columns(
        df = data,
        columns = treatment_col,
        verbose = verbose
        )
    
    tools.is_numeric(
        df = data,
        columns = treatment_col,
        verbose = verbose
        )
    
    cols_relevant = [
        unit_col,
        time_col,
        *treatment_col        
        ]
    
    data = tools.panel_index(
        data = data,
        unit_col = unit_col,
        time_col = time_col,
        verbose = verbose
        )
    
    treatment_diagnostics_results = helper.treatment_diagnostics(
        data = data,
        unit_col=unit_col,
        time_col=time_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        pre_post=pre_post,
        confint_alpha=confint_alpha,
        verbose=verbose
    )
    treatment_diagnostics = treatment_diagnostics_results[0]
    staggered_adoption = treatment_diagnostics_results[1]
    
    if no_treatments > 1:
        FE_unit = True
        intercept = False
        TG_col = []
        print("NOTE: Quasi-experiment includes more than one treatment. Unit fixed effects are used instead of control group baseline and treatment group deviation.")
           
    if ITE:
        
        FE_unit = True
        print("NOTE: Model includes individual treatment effects. Unit fixed effects are included.")
        
        if GTE:
            GTE = False
            print("NOTE: Both group and individual treatment effects were stated. Switching to individual treatment effects only.")
    
    if ITT:
        
        FE_unit = True
        
        TT_col = []
        
        print("NOTE: Model includes individual time trends. Unit fixed effects are included. Treatment time variable is dropped.")
        
        if FE_time:
            FE_time = False
            print("NOTE: Time fixed effects are dropped.")
        
        if GTT:
            GTT = False
            print("NOTE: Both group and individual time trends were stated. Switching to individual time trends only.")
            
    if staggered_adoption:
        
        FE_unit = True
        FE_time = True        
        
        print("NOTE: Quasi-experiment includes one or more staggered treatments. Two-way fixed effects model is used.")
    
    if GTT or GTE:        
        FE_group = True
               
    if FE_unit:
        TG_col = []
        
    if FE_time:
        TT_col = []
        
    if FE_group:    
        TG_col = []        
        intercept = False
        print("NOTE: Quasi-experiment includes group fixed effects. Control group baseline and treatment group deviation are dropped.")   
    
    if after_treatment_col is not None or (isinstance (after_treatment_col, list) and len(after_treatment_col) > 0):

        if isinstance (after_treatment_col, str):
            after_treatment_col = [after_treatment_col]
        after_treatment_col = [entry for entry in after_treatment_col if entry is not None]
        
        tools.check_columns(
            df = data,
            columns = after_treatment_col,
            verbose = verbose
            )        
        
        cols_relevant = cols_relevant + after_treatment_col

    if ATT_col is not None or (isinstance (ATT_col, list) and len(ATT_col) > 0):
        
        if isinstance (ATT_col, str):
            ATT_col = [ATT_col]
        ATT_col = [entry for entry in ATT_col if entry is not None]
        
        tools.check_columns(
            df = data,
            columns = ATT_col,
            verbose = verbose
            )        
        
        cols_relevant = cols_relevant + ATT_col

    if TG_col is not None or (isinstance (TG_col, list) and len(TG_col) > 0):
        
        if isinstance (TG_col, str):
            TG_col = [TG_col]
        TG_col = [entry for entry in TG_col if entry is not None]        
        
        tools.check_columns(
            df = data,
            columns = TG_col,
            verbose = verbose
            )        
        
        cols_relevant = cols_relevant + TG_col        
    else:

        FE_unit = True

    if TT_col is not None or (isinstance (TT_col, list) and len(TT_col) > 0):

        if isinstance (TT_col, str):
            TT_col = [TT_col]
        TT_col = [entry for entry in TT_col if entry is not None]

        tools.check_columns(
            df = data,
            columns = TT_col,
            verbose = verbose
            )
          
        cols_relevant = cols_relevant + TT_col
    else:        
        FE_time = True

    if covariates is not None or (isinstance (covariates, list) and len(covariates) > 0):

        tools.check_columns(
            df = data,
            columns = covariates,
            verbose = verbose
            )
          
        cols_relevant = cols_relevant + covariates

    if group_by is not None and group_by != "":

        tools.check_columns(
            df = data,
            columns = [group_by],
            verbose = verbose
            )
        
        if group_by not in data.columns:
            cols_relevant = cols_relevant + [group_by]   
    
    cols_relevant = cols_relevant + [outcome_col]
    data = data[cols_relevant].copy()
            
    data = tools.date_counter(
        data,
        time_col,
        new_col = config.TIME_COUNTER_COL,
        verbose = verbose
        )

    data_diagnostics = helper.data_diagnostics(
        data = data,
        unit_col = unit_col,
        time_col = time_col,
        outcome_col = outcome_col,
        cols_relevant = cols_relevant,
        drop_missing = drop_missing,
        missing_replace_by_zero = missing_replace_by_zero,
        verbose = verbose
    )
        
    if data_diagnostics["is_prepost"] and config.AUTO_SWITCH_TO_PREPOST:
        
        print("NOTE: Panel data is pre-post. Data processing and model estimation will treat data as pre-post")
        
        pre_post = True
        
    if log_outcome:
        
        if missing_replace_by_zero:
            data[f"{config.LOG_PREFIX}{config.DELIMITER}{outcome_col}"] = np.log(data[outcome_col]+log_outcome_add)
        
        else:
            data[f"{config.LOG_PREFIX}{config.DELIMITER}{outcome_col}"] = np.log(data[outcome_col])
        
        outcome_col = f"{config.LOG_PREFIX}{config.DELIMITER}{outcome_col}"

    if not ITE and not GTE:
        did_formula = f"{outcome_col} ~ {' + '.join(treatment_col)}"
    else:
        did_formula = f"{outcome_col} ~ "        
    
    if TG_col is not None and len(TG_col) > 0:
        did_formula = f"{did_formula} + {' + '.join(TG_col)}"
        
    if TT_col is not None and len(TT_col) > 0:
        did_formula = f"{did_formula} + {' + '.join(TT_col)}"

    if len(after_treatment_col) > 0:
        did_formula = f"{did_formula} + {' + '.join(after_treatment_col)}"
        
    if len(ATT_col) > 0:
        did_formula = f"{did_formula} + {' + '.join(ATT_col)}"    

    indep_vars_no = len(treatment_col)+len(TG_col)+len(TT_col)+len(after_treatment_col)+len(ATT_col)

    FE_unit_vars = []
    dummy_unit_original = []

    if FE_unit:

        FE_unit_dummies = helper.create_fixed_effects(
            data = data,
            col = unit_col,
            type = "unit",
            drop_first = intercept,
            verbose = verbose
        )
        
        data = FE_unit_dummies[0]        
        did_formula = f"{did_formula} + {FE_unit_dummies[1]}"
        FE_unit_vars = [col for col in FE_unit_dummies[2] if col in data.columns]
        dummy_unit_original = FE_unit_dummies[3]

    FE_time_vars = []
    dummy_time_original = []

    if FE_time:
        
        FE_time_dummies = helper.create_fixed_effects(
            data = data,
            col = time_col,
            type = "time",
            drop_first = intercept,
            verbose = verbose
        )
        
        data = FE_time_dummies[0]
        did_formula = f"{did_formula} + {FE_time_dummies[1]}"
        FE_time_vars = [col for col in FE_time_dummies[2] if col in data.columns]
        dummy_time_original = FE_time_dummies[3] 

    FE_group_vars = []
    dummy_group_original = []

    if FE_group:
        
        if group_by is None or group_by == "":
        
            print("WARNING: Grouping variable is not defined. No group-specific analyses are carried out. Define a grouping variable using group_by.")
        
        else:
            
            FE_group_dummies = helper.create_fixed_effects(
                data = data,
                col = group_by,
                type = "group",
                drop_first = intercept,
                verbose = verbose
                )
        
            data = FE_group_dummies[0]
            did_formula = f"{did_formula} + {FE_group_dummies[1]}"
            FE_group_vars = [col for col in FE_group_dummies[2] if col in data.columns]
            dummy_group_original = FE_group_dummies[3]      
    
    indep_vars_no = indep_vars_no+len(FE_unit_vars)+len(FE_time_vars)+len(FE_group_vars)
    
    GTT_vars = []
    
    if GTT and group_by is not None and group_by != "":            
         
        time_trend_group_vars = helper.create_specific_time_trends(
            data = data,
            time_col = time_col,
            FE_vars = FE_group_vars,
            verbose = verbose
        )

        data = time_trend_group_vars[0]
        did_formula = f"{did_formula} + {time_trend_group_vars[1]}"
        GTT_vars = time_trend_group_vars[2]
                 
    ITT_vars = []
    
    if ITT:
        
        time_trend_indiv_vars = helper.create_specific_time_trends(
            data = data,
            time_col = time_col,
            FE_vars = FE_unit_vars,
            verbose = verbose
        )

        data = time_trend_indiv_vars[0]
        did_formula = f"{did_formula} + {time_trend_indiv_vars[1]}"
        ITT_vars = time_trend_indiv_vars[2]
        
    GTE_vars = []

    if GTE and group_by is not None and group_by != "":
         
        group_treatment_vars = helper.create_specific_treatment_effects(
            data = data,
            treatment_col = treatment_col,
            FE_vars = FE_group_vars,
            type = "GTE",
            verbose = verbose
        )

        data = group_treatment_vars[0]
        did_formula = f"{did_formula} + {group_treatment_vars[1]}"
        GTE_vars = group_treatment_vars[2]
        
    ITE_vars = []

    if ITE:

        indiv_treatment_vars = helper.create_specific_treatment_effects(
            data = data,
            treatment_col = treatment_col,
            FE_vars = FE_unit_vars,
            type = "ITE",
            verbose = verbose
        )

        data = indiv_treatment_vars[0]
        did_formula = f"{did_formula} + {indiv_treatment_vars[1]}"
        ITE_vars = indiv_treatment_vars[2]
        
    indep_vars_no = indep_vars_no+len(GTT_vars)+len(ITT_vars)+len(GTE_vars)+len(ITE_vars)

    spillover_effects = False
    spillover_vars = []

    if spillover_treatment is not None and len(spillover_treatment) > 0 and spillover_units is not None and len(spillover_units) > 0:
        
        spillover_effects = True

        spillover = helper.create_spillover(
            data=data,
            unit_col=unit_col,
            time_col=time_col,
            treatment_col=treatment_col,
            spillover_treatment=spillover_treatment,
            spillover_units=spillover_units
        )

        data = spillover[0]

        did_formula = f"{did_formula} + {spillover[1]}"
        spillover_vars = spillover[2]

    if len(covariates) > 0:

        if group_by in covariates:
            covariates.remove(group_by)
        
        covariates_join = ' + '.join(covariates)
        
        did_formula = f"{did_formula} + {covariates_join}"
        
    indep_vars_no = indep_vars_no+len(spillover_vars)+len(covariates)
        
    did_formula = did_formula[:-1] if did_formula.endswith(" ") else did_formula
    did_formula = did_formula[:-1] if did_formula.endswith("+") else did_formula
    did_formula = did_formula[:-1] if did_formula.endswith(" ") else did_formula
    if not intercept:
        did_formula = f"{did_formula} - 1"
            
    analysis_description = config.DID_DESCRIPTION
    if placebo:
        analysis_description = f"Placebo {config.DID_DESCRIPTION}"

    model_config = {
        "TG_col": TG_col,
        "TT_col": TT_col,
        "treatment_col": treatment_col,
        "unit_col": unit_col,
        "time_col": time_col,
        "outcome_col": outcome_col,
        "log_outcome": log_outcome,
        "freq": freq,
        "date_format": date_format,
        "after_treatment_col": after_treatment_col,
        "ATT_col": ATT_col,
        "pre_post": pre_post,
        "FE_unit": FE_unit,
        "FE_time": FE_time,
        "FE_group": FE_group,
        "cluster_SE_by": cluster_SE_by,
        "intercept": intercept,
        "ITT": ITT,
        "GTT": GTT,
        "ITE": ITE,
        "GTE": GTE,
        "group_by": group_by,
        "covariates": covariates,
        "spillover_effects": spillover_effects,
        "placebo": placebo,
        "confint_alpha": confint_alpha,
        "bonferroni": bonferroni,
        "drop_missing": drop_missing,
        "no_treatments": no_treatments,
        "treatment_diagnostics": treatment_diagnostics,
        "data_diagnostics": data_diagnostics,
        "did_formula": did_formula,
        "analysis_description": analysis_description,
        "fit_by": fit_by,
        "indep_vars_no": indep_vars_no
        }
    
    if bonferroni:
        confint_alpha = confint_alpha/no_treatments

    if fit_by == "ml":
        fit_result = helper.ml_fit(
            data = data,
            formula = did_formula,
            confint_alpha = confint_alpha,
            verbose = verbose
        )
    else:
        fit_result = helper.ols_fit(
            data = data,
            formula = did_formula,
            confint_alpha = confint_alpha,
            cluster_SE_by = cluster_SE_by,
            verbose = verbose
        )
           
    model_results = helper.extract_model_results(
        fit_result = fit_result,
        TG_col = TG_col,
        TT_col = TT_col,
        treatment_col = treatment_col,
        after_treatment_col = after_treatment_col,
        ATT_col = ATT_col,
        spillover_vars = spillover_vars,
        FE_unit_vars = FE_unit_vars,
        dummy_unit_original = dummy_unit_original,
        FE_time_vars = FE_time_vars,
        dummy_time_original = dummy_time_original,
        FE_group_vars = FE_group_vars,
        dummy_group_original = dummy_group_original,
        ITE_vars = ITE_vars,
        GTE_vars = GTE_vars,
        ITT_vars = ITT_vars,
        GTT_vars = GTT_vars,
        TG_x_BG_x_TT_col = [],
        BG_col = [], 
        TG_x_BG_col = [], 
        BG_x_TT_col = [],
        covariates = covariates,
        verbose = verbose
        )

    did_model = fit_result[0]
    model_predictions = fit_result[6]
    
    did_model_output = DiffModel(
        model_results,
        model_config,
        data,
        model_predictions,
        did_model,
        timestamp = helper.create_timestamp(function="did_analysis")
        )
    
    return did_model_output

def ddd_analysis(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    TG_col: str,
    TT_col: str,
    BG_col: str,
    pre_post: bool = False,
    log_outcome: bool = False,
    log_outcome_add = 0.01,
    FE_unit: bool = False,
    FE_time: bool = False,
    covariates: list = None,
    placebo: bool = False,
    confint_alpha = 0.05,
    freq: str = "D",
    date_format: str = "%Y-%m-%d",
    drop_missing: bool = True,
    missing_replace_by_zero: bool = False,
    fit_by: str = "ols_fit",
    verbose: bool = config.VERBOSE
    ):
    
    """
    Perform a Difference-in-difference-in-differences (DDD) analysis
    with a given data frame.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data containing units, time and outcome.
    unit_col : str
        Name of unit identifier column.
    time_col : str
        Name of time identifier column.
    outcome_col : str
        Name of outcome variable column.
    TG_col : str
        Treatment group indicator column.
    TT_col : str
        Treatment time indicator column.
    BG_col : str
        Benefit/group identifier column.
    pre_post : bool
        Treat data as pre-post panel. Default is False.
    log_outcome : bool
        Log-transform the outcome. Default is False.
    log_outcome_add : float
        Constant added before log-transform when zeros present. Default is 0.01.
    FE_unit : bool
        Include unit fixed effects. Default is False.
    FE_time : bool
        Include time fixed effects. Default is False.
    covariates : list
        Additional covariate column names to include. Default is None.
    placebo : bool
        Run placebo analysis. Default is False.
    confint_alpha : float
        Significance level for confidence intervals. Default is 0.05.
    freq : str
        Frequency string for date handling. Default is 'D'.
    date_format : str
        Date format string. Default is '%Y-%m-%d'.
    drop_missing : bool
        Drop missing observations before analysis. Default is True.
    missing_replace_by_zero : bool
        Replace missing values by zero when requested. Default is False.
    fit_by : str
        Fitting method; e.g., 'ols_fit' or 'ml'. Default is 'ols_fit'.
    verbose : bool
        If True, print progress messages.

    Returns
    -------
    DiffModel
        Container with DDD model results, configuration, data and helpers.

    Raises
    ------
    ValueError
        If required columns are missing or invalid parameters are provided.

    Examples
    --------
    >>> curfew_DE=pd.read_csv("data/curfew_DE.csv", sep=";", decimal=",")
    >>> ddd_analysis(
    ...     data=curfew_DE,
    ...     unit_col="county",
    ...     time_col="infection_date",
    ...     outcome_col="infections_cum_per100000",
    ...     TG_col="TG",
    ...     TT_col="TT",
    ...     BG_col="some_benefit_group_variable",
    ... )
    """

    tools.check_columns(
        df = data,
        columns = [
            unit_col, 
            time_col, 
            outcome_col,
            TG_col,
            TT_col,
            BG_col
            ],
        verbose = verbose
        )
    
    TG_x_BG_col = f"{TG_col}{config.DELIMITER_INTERACT}{BG_col}"
    TG_x_TT_col = f"{TG_col}{config.DELIMITER_INTERACT}{TT_col}"
    BG_x_TT_col = f"{BG_col}{config.DELIMITER_INTERACT}{TT_col}"
    TG_x_BG_x_TT_col = f"{TG_col}{config.DELIMITER_INTERACT}{BG_col}{config.DELIMITER_INTERACT}{TT_col}"
    
    data[TG_x_BG_col] = data[TG_col].astype(str) + config.DELIMITER_INTERACT + data[BG_col].astype(str)
    data[TG_x_TT_col] = data[TG_col].astype(str) + config.DELIMITER_INTERACT + data[TT_col].astype(str)
    data[BG_x_TT_col] = data[BG_col].astype(str) + config.DELIMITER_INTERACT + data[TT_col].astype(str)
    data[TG_x_BG_x_TT_col] = data[TG_col].astype(str) + config.DELIMITER_INTERACT + data[BG_col].astype(str) + config.DELIMITER_INTERACT + data[TT_col].astype(str)
    
    cols_relevant = [
        unit_col,
        time_col,
        outcome_col,
        TG_col,
        TT_col,
        BG_col,
        TG_x_BG_col,
        TG_x_TT_col,
        BG_x_TT_col,
        TG_x_BG_x_TT_col
        ]   
    
    if (isinstance (covariates, list) and len(covariates) > 0):

        tools.check_columns(
            df = data,
            columns = covariates,
            verbose = verbose
            )
          
        cols_relevant = cols_relevant + covariates
    
    data = tools.panel_index(
        data = data,
        unit_col = unit_col,
        time_col = time_col,
        verbose = verbose
        )
    
    treatment_diagnostics_results = helper.treatment_diagnostics(
        data = data,
        unit_col=unit_col,
        time_col=time_col,
        treatment_col=[TG_x_TT_col],
        outcome_col=outcome_col,
        pre_post=True,
        confint_alpha=confint_alpha,
        verbose=verbose
    )
    treatment_diagnostics = treatment_diagnostics_results[0]
               
    if FE_unit:
        TG_col = None
    if FE_time:
        TT_col = None
        
    data = data[cols_relevant].copy()
            
    data = tools.date_counter(
        data,
        time_col,
        new_col = config.TIME_COUNTER_COL,
        verbose = verbose
        )

    data_diagnostics = helper.data_diagnostics(
        data = data,
        unit_col = unit_col,
        time_col = time_col,
        outcome_col = outcome_col,
        cols_relevant = cols_relevant,
        drop_missing = drop_missing,
        missing_replace_by_zero = missing_replace_by_zero,
        verbose = verbose
    )
      
    if data_diagnostics["is_prepost"] and config.AUTO_SWITCH_TO_PREPOST:
        
        print("NOTE: Panel data is pre-post. Data processing and model estimation will treat data as pre-post")
        
        pre_post = True
        
    if log_outcome:
        
        if missing_replace_by_zero:
            data[f"{config.LOG_PREFIX}{config.DELIMITER}{outcome_col}"] = np.log(data[outcome_col]+log_outcome_add)
        
        else:
            data[f"{config.LOG_PREFIX}{config.DELIMITER}{outcome_col}"] = np.log(data[outcome_col])
        
        outcome_col = f"{config.LOG_PREFIX}{config.DELIMITER}{outcome_col}"

    ddd_formula = f"{outcome_col} ~ {BG_col} + {TG_x_BG_col} + {TG_x_TT_col} + {BG_x_TT_col} + {TG_x_BG_x_TT_col}"
        
    if TG_col is not None:
        ddd_formula = f"{ddd_formula} + {TG_col}"
    if TT_col is not None:
        ddd_formula = f"{ddd_formula} + {TT_col}"

    FE_unit_vars = []
    
    if FE_unit:

        FE_unit_dummies = helper.create_fixed_effects(
            data = data,
            col = unit_col,
            type = "unit",
            drop_first = False,
            verbose = verbose
        )
        
        data = FE_unit_dummies[0]        
        ddd_formula = f"{ddd_formula} + {FE_unit_dummies[1]}"
        FE_unit_vars = [col for col in FE_unit_dummies[2] if col in data.columns]
        dummy_unit_original = FE_unit_dummies[3]

    FE_time_vars = []
    
    if FE_time:
        
        FE_time_dummies = helper.create_fixed_effects(
            data = data,
            col = time_col,
            type = "time",
            drop_first = False,
            verbose = verbose
        )
        
        data = FE_time_dummies[0]
        ddd_formula = f"{ddd_formula} + {FE_time_dummies[1]}"
        FE_time_vars = [col for col in FE_time_dummies[2] if col in data.columns]
        dummy_time_original = FE_time_dummies[3] 

    if len(covariates) > 0:

        covariates_join = ' + '.join(covariates)
        
        ddd_formula = f"{ddd_formula} + {covariates_join}"
        
    ddd_formula = ddd_formula[:-1] if ddd_formula.endswith(" ") else ddd_formula
    ddd_formula = ddd_formula[:-1] if ddd_formula.endswith("+") else ddd_formula
    ddd_formula = ddd_formula[:-1] if ddd_formula.endswith(" ") else ddd_formula
    
    analysis_description = config.DDD_DESCRIPTION
    
    if placebo:
        analysis_description = f"Placebo {config.DDD_DESCRIPTION}"

    model_config = {
        "TG_col": TG_col,
        "TT_col": TT_col,
        "BG_col": BG_col,
        "treatment_col": [TG_x_TT_col],
        "unit_col": unit_col,
        "time_col": time_col,
        "outcome_col": outcome_col,
        "log_outcome": log_outcome,
        "freq": freq,
        "date_format": date_format,
        "after_treatment_col": None,
        "ATT_col": None,
        "pre_post": pre_post,
        "FE_unit": FE_unit,
        "FE_time": FE_time,
        "FE_group": None,
        "cluster_SE_by": None,
        "intercept": True,
        "ITT": False,
        "GTT": False,
        "ITE": False,
        "GTE": False,
        "group_by": None,
        "covariates": covariates,
        "DDD": True,
        "spillover_effects": None,
        "placebo": placebo,
        "confint_alpha": confint_alpha,
        "bonferroni": False,
        "drop_missing": drop_missing,
        "no_treatments": 1,
        "treatment_diagnostics": treatment_diagnostics,
        "data_diagnostics": data_diagnostics,
        "did_formula": ddd_formula,
        "analysis_description": analysis_description,
        "fit_by": fit_by
        }
   
    if fit_by == "ml":
        fit_result = helper.ml_fit(
            data = data,
            formula = ddd_formula,
            confint_alpha = confint_alpha,
            verbose = verbose
        )
    else:
        fit_result = helper.ols_fit(
            data = data,
            formula = ddd_formula,
            confint_alpha = confint_alpha,
            verbose = verbose
        )
    
    model_results = helper.extract_model_results(
        fit_result = fit_result,
        TG_col = TG_col,
        TT_col = TT_col,
        treatment_col = [TG_x_TT_col],
        after_treatment_col = None,
        ATT_col = None,
        spillover_vars = None,
        FE_unit_vars = FE_unit_vars,
        dummy_unit_original = dummy_unit_original,
        FE_time_vars = FE_time_vars,
        dummy_time_original = dummy_time_original,
        FE_group_vars = None,
        dummy_group_original = None,
        ITE_vars = None,
        GTE_vars = None,
        ITT_vars = None,
        GTT_vars = None,
        TG_x_BG_x_TT_col = [TG_x_BG_x_TT_col],
        BG_col = [BG_col], 
        TG_x_BG_col = [TG_x_BG_col], 
        BG_x_TT_col = [BG_x_TT_col],
        covariates = covariates,
        verbose = verbose
        )

    did_model = fit_result[0]
    model_predictions = fit_result[6]
    
    did_model_output = DiffModel(
        model_results,
        model_config,
        data,
        model_predictions,
        did_model,
        timestamp = helper.create_timestamp(function="ddd_analysis")
        )
    
    if verbose:
        print("OK")

    return did_model_output