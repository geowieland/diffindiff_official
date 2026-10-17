#-----------------------------------------------------------------------
# Name:        tests (diffindiff package)
# Purpose:     Tests for diffindiff package functions
# Author:      Thomas Wieland 
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com              
# Version:     1.0.0
# Last update: 2026-10-17 15:01
# Copyright (c) 2026 Thomas Wieland
#-----------------------------------------------------------------------


from pathlib import Path
import pandas as pd


def test_import_diffindiff():
    """Test if diffindiff package can be imported."""

    import diffindiff
    from diffindiff.config import PACKAGE_NAME, PACKAGE_VERSION

    print(f"This is the {PACKAGE_NAME} package v{PACKAGE_VERSION}")


def test_prepost_analysis():

    from diffindiff.diddata import create_groups, create_treatment, merge_data, create_data
   
    curfew_DE = pd.read_csv(
        Path(__file__).parent / "data" / "curfew_DE.csv",
        sep=";",
        decimal=","
    )
    # Dataset with daily and cumulative SARS-CoV-2 infections of German counties
    # Data source: Wieland (2020) https://doi.org/10.18335/region.v7i2.324

    curfew_groups=create_groups(
        treatment_group= 
            curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        control_group= 
            curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"]
        )
    # Creating treatment and control group
    # "Bundesland" (=federal state): 9 = Bavaria, 10 = Saarland, 14 = Saxony

    assert curfew_groups is not None

    curfew_groups.summary()
    # Groups summary

    curfew_treatment_prepost=create_treatment(
        study_period=["2020-03-01", "2020-05-15"],
        treatment_period=["2020-03-21", "2020-05-05"],
        freq = "D",
        pre_post = True
        )
    # Creating treatment
    # Curfew from March 21, 2020, to May, 5, 2020

    assert curfew_treatment_prepost is not None

    curfew_treatment_prepost.summary()
    # Treatment summary

    curfew_data_prepost_merge=merge_data(
        outcome_data=curfew_DE,
        unit_id_col="county",
        time_col="infection_date",
        outcome_col="infections_cum_per100000",
        diff_groups=curfew_groups,
        diff_treatment=curfew_treatment_prepost
        )

    assert curfew_data_prepost_merge is not None

    curfew_data_prepost_merge.summary()
    # Summary of created data

    curfew_data_prepost=create_data(
        outcome_data=curfew_DE,
        unit_id_col="county",
        time_col="infection_date",
        outcome_col="infections_cum_per100000",
        treatment_group= 
            curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        control_group= 
            curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        study_period=["2020-03-01", "2020-05-15"],
        treatment_period=["2020-03-21", "2020-05-05"],
        freq="D",
        pre_post=True,
        after_treatment_period = False
        )
    # Creating DiD treatement dataset by defining groups and
    # treatment time at once

    assert curfew_data_prepost is not None

    curfew_data_prepost.summary()
    # Summary of created data

    curfew_model_prepost=curfew_data_prepost.analysis(verbose=True)
    # Model analysis of created data

    assert curfew_model_prepost is not None

    print(curfew_model_prepost.treatment_effects())
    # Show treatment effects

    print(curfew_model_prepost.fixed_effects())
    # Show fixed effects (not included in model)

    print(curfew_model_prepost.covariates())
    # Show covariates (not included in model)

    curfew_model_prepost.summary()
    # Model summary

    print(curfew_model_prepost.didmodel().summary())
    # Access statsmodels OLS object

    print(curfew_model_prepost.fit_metrics())
    # Show model fit metrics

    curfew_model_prepost.plot(
        x_label="Timepoint",
        y_label="Cumulative infections per 100,000",
        plot_title="Curfew effectiveness pre-post - Groups over time",
        plot_observed=False,
        lines_col=[None,None,"blue","orange"],
        lines_labels=[None,None,"Treatment group","Control group","Treatment group CI","Control group CI"],
        lines_style=[None,None,"solid","solid"]
        )
    # Plot DiD pre vs. post results
    # with user-determined style

    curfew_model_prepost.plot(
        x_label="Timepoint",
        y_label="Cumulative infections per 100,000",
        plot_title="Curfew effectiveness pre-post - Groups over time",
        lines_col=[None,None,"blue","orange"],
        lines_labels=[None,None,"Treatment group","Control group","Treatment group CI", "Control group CI"],
        pre_post_barplot=True
        )
    # Plot DiD pre vs. post results
    # with user-determined style

    curfew_model_prepost.plot_treatment_effects(
        x_label="Coefficients with 95% CI",
        plot_title="Curfew effectiveness pre-post - DiD effects"
        )
    # plot effects

    curfew_model_prepost.plot_treatment_effects(
        x_label="Coefficients with 95% CI",
        plot_title="Curfew effectiveness pre-post - DiD effects",
        scale_plot=False
        )
    # plot effects

def test_multiperiod_paneldata_analysis():

    from diffindiff.diddata import create_data
       
    curfew_DE = pd.read_csv(
        Path(__file__).parent / "data" / "curfew_DE.csv",
        sep=";",
        decimal=","
    )
    # Dataset with daily and cumulative SARS-CoV-2 infections of German counties
    # Data source: Wieland (2020) https://doi.org/10.18335/region.v7i2.324

    curfew_data=create_data(
        outcome_data=curfew_DE,
        unit_id_col="county",
        time_col="infection_date",
        outcome_col="infections_cum_per100000",
        treatment_group= 
            curfew_DE.loc[curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        control_group= 
            curfew_DE.loc[~curfew_DE["Bundesland"].isin([9,10,14])]["county"],
        treatment_name="Curfew",
        study_period=["2020-03-01", "2020-05-15"],
        treatment_period=["2020-03-21", "2020-05-05"],
        freq="D"
        )
    # Creating DiD dataset by defining groups and treatment time at once
    # Treatment curfew in three federal states: Bavaria (9), Saarland (10) and Saxony (14)
   
    assert curfew_data is not None

    curfew_data.summary()
    # Summary of created treatment data

    curfew_model=curfew_data.analysis(verbose=True)
    # Model analysis of created data

    assert curfew_model is not None

    curfew_model.summary()
    # Model summary

    curfew_model.plot(
        y_label="Cumulative infections per 100,000",
        plot_title="Curfew effectiveness - Groups over time",
        plot_observed=True
        )
    # Plot observed vs. predicted (means) separated by group (treatment and control)

    curfew_model.plot_treatment_effects(
        x_label="Coefficients with 95% CI",
        plot_title="Curfew effectiveness - DiD effects"
        )
    # plot effects

    curfew_placebo = curfew_model.placebo(
        treatment="Curfew"
        )
    # Placebo test with default parameters

    assert curfew_placebo is not None

    curfew_placebo.summary()
    # Summary of placebo test

    # Two-way-fixed-effects model:

    curfew_model_FE=curfew_data.analysis(
        FE_unit=True, 
        FE_time=True,
        demean = True,
        verbose = True
        )
    # Model analysis of created data with fixed effects for 
    # units (FE_unit=True) and time (FE_time=True)
    # data is demeaned

    assert curfew_model_FE is not None

    curfew_model_FE_summary = curfew_model_FE.summary()
    # Model summary

    assert curfew_model_FE_summary is not None

    print(curfew_model_FE.treatment_statistics())
    # Print treatment statistics

    curfew_model_FE.plot(
        y_label="Cumulative infections per 100,000",
        plot_title="Curfew effectiveness (Two-way FE model) - Groups over time",
        plot_observed=True
        )
    # Plot of treatment and control group

    fixed_effects = curfew_model_FE.fixed_effects()
    # Fixed effects of CHBW_data_model_FE