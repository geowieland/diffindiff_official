# diffindiff: Python library for convenient Difference-in-Differences analyses

This Python library is designed for performing Difference-in-Differences (DiD) analyses in a convenient way. The package is intended to be used in econometric analyses of natural experiments by researchers in economics, marketing, economic geography, and health sciences. It is designed to cover the entire workflow of a DiD analysis, while not requiring extensive programming skills. The package allows users to construct datasets, define treatment and control groups, and set treatment periods. DiD model analyses may be conducted with both datasets created by built-in functions and ready-to-use external datasets. Both simultaneous and staggered adoption are supported. The library allows for various extensions, such as two-way fixed effects models, group- or individual-specific effects, post-treatment periods, and triple-difference estimations. Additionally, it includes functions for visualizing results, such as plotting DiD coefficients with confidence intervals and illustrating the temporal evolution of staggered treatments. Furthermore, several functions for rigorous treatment setting and data diagnostics are incorporated.


## Author

Thomas Wieland [ORCID](https://orcid.org/0000-0001-5168-9846) [EMail](mailto:geowieland@googlemail.com) 


## Availability

- 📦 PyPI: [diffindiff](https://pypi.org/project/diffindiff/)
- 💻 GitHub Repository: [diffindiff_official](https://github.com/geowieland/diffindiff_official)
- 📄 DOI (Zenodo): [10.5281/zenodo.18656820](https://doi.org/10.5281/zenodo.18656820)


## Citation

If you use this software, please cite:

Wieland, T. (2026). diffindiff: A Python library for convenient difference-in-differences analyses (Version 2.3.4) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.18656820


## Installation

To install the package, use `pip`:

```bash
pip install diffindiff
```

To install the package from GitHub with `pip`: 

```bash
pip install git+https://github.com/geowieland/diffindiff_official.git
```


## Features

- **Data preparation and pre-analysis**: 
  - Define custom treatment and control groups as well as treatment periods
  - Create ready-to-fit DiD data objects
  - Create predictive counterfactuals
- **DiD analysis**: 
  - Perform standard DiD analysis with pre-post data
  - Perform DiD analysis with two-way fixed effects models
  - Simultaneous and/or staggered adoption are supported
  - Single or multiple treatments are supported
  - Binary or continuous treatments are supported
  - Model extensions for DiD analysis:
    - Group- or individual-specific treatment effects
    - Group- or individual-specific time trends
    - Including covariates
    - Including after-treatment period
  - Perform Triple Difference (DDD) analysis
- **Diagnosis tools and extensions of analyses**:
  - Add own counterfactuals or create counterfactuals based on machine learning or OLS regression models
  - Bonferroni correction for treatment effects
  - Placebo test
  - Test for control conditions (automatically within analysis or stand-alone)
  - Test for type of adoption (automatically within analysis or stand-alone)
  - Test whether the panel dataset is balanced (automatically within analysis or stand-alone)
  - Test for parallel trend assumption (automatically within analysis or stand-alone)
- **Visualization**:
  - Plot observed and expected time course of treatment and control group
  - Plot expected time course of treatment group and counterfactual
  - Plot model coefficients with confidence intervals
  - Plot individual or group-specific treatment effects with confidence intervals
  - Visualize the temporal evolution of staggered treatments


## Examples

```python
curfew_DE=pd.read_csv("data/curfew_DE.csv", sep=";", decimal=",")
# Test dataset: Daily and cumulative COVID-19 infections in German counties

curfew_data=create_data(
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
    freq="D"
    )
# Creating DiD dataset by defining groups and treatment time

curfew_data.summary()
# Summary of created treatment data

curfew_model = curfew_data.analysis()
# Model analysis of created data

curfew_model.summary()
# Model summary

curfew_model.plot(
    y_label="Cumulative infections per 100,000",
    plot_title="Curfew effectiveness - Groups over time",
    plot_observed=True
    )
# Plot observed vs. predicted (means) separated by group (treatment and control)

curfew_model.plot_effects(
    x_label="Coefficients with 95% CI",
    plot_title="Curfew effectiveness - DiD effects"
    )
# plot effects

counties_DE=pd.read_csv("data/counties_DE.csv", sep=";", decimal=",", encoding='latin1')
# Dataset with German county data

curfew_data_withgroups = curfew_data.add_covariates(
    additional_df=counties_DE, 
    unit_col="county",
    time_col=None, 
    variables=["BL"])
# Adding federal state column as covariate

curfew_model_withgroups = curfew_data_withgroups.analysis(
    GTE=True,
    group_by="BL")
# Model analysis of created data

curfew_model_withgroups.summary()
# Model summary

curfew_model_withgroups.plot_group_treatment_effects(
    treatment_group_only=True
    )
# Plot of group-specific treatment effects
```

See the /tests directory for usage examples of most of the included functions.


## Literature

  - Baker AC, Larcker DF, Wang CCY (2022) How much should we trust staggered difference-in-differences estimates? *Journal of Financial Economics* 144(2): 370-395. [10.1016/j.jfineco.2022.01.004](https://doi.org/10.1016/j.jfineco.2022.01.004)
  - Card D, Krueger AD (1994) Minimum Wages and Employment: A Case Study of the Fast Food Industry in New Jersey and Pennsylvania. *The American Economic Review* 84(4): 772-793. [JSTOR](https://www.jstor.org/stable/2677856)
  - de Haas S, Götz G, Heim S (2022) Measuring the effect of COVID‑19‑related night curfews in a bundled intervention within Germany. *Scientific Reports* 12: 19732. [10.1038/s41598-022-24086-9](https://doi.org/10.1038/s41598-022-24086-9)
  - Goodman-Bacon A (2021) Difference-in-differences with variation in treatment timing. *Journal of Econometrics* 225(2): 254-277. [10.1016/j.jeconom.2021.03.014](https://doi.org/10.1016/j.jeconom.2021.03.014)
  - Greene WH (2012) *Econometric Analysis*.
  - Goldfarb A, Tucker C, Wang Y (2022) Conducting Research in Marketing with Quasi-Experiments. *Journal of Marketing* 86(3): 1-19. [10.1177/00222429221082977](https://doi.org/10.1177/00222429221082977)
  - Isporhing IE, Lipfert M, Pestel N (2021) Does re-opening schools contribute to the spread of SARS-CoV-2? Evidence from staggered summer breaks in Germany. *Journal of Public Economics* 198: 104426. [10.1016/j.jpubeco.2021.104426](https://doi.org/10.1016/j.jpubeco.2021.104426)
  - Li KT, Luo L, Pattabhiramaiah A (2024) Causal Inference with Quasi-Experimental Data. *IMPACT at JMR* November 13, 2024. [AMA](https://www.ama.org/marketing-news/causal-inference-with-quasi-experimental-data/)
  - Olden A (2018) What do you buy when no one's watching? The effect of self-service checkouts on the composition of sales in retail. Discussion paper FOR 3/18, Norwegian School of Economics, Norway. [http://hdl.handle.net/11250/2490886](http://hdl.handle.net/11250/2490886)
  - Olden A, Moen J (2022) The triple difference estimator. *The Econometrics Journal* 25(3): 531-553. [10.1093/ectj/utac010](https://doi.org/10.1093/ectj/utac010)
  - Strassmann A, Çolak Y, Serra-Burriel M, Nordestgaard BG, Turk A, Afzal S, Puhan MA (2023) Nationwide indoor smoking ban and impact on smoking behaviour and lung function: a two-population natural experiment. *Thorax* 78(2): 144-150. [10.1136/thoraxjnl-2021-218436](https://doi.org/10.1136/thoraxjnl-2021-218436)
  - Villa JM (2016) diff: Simplifying the estimation of difference-in-differences treatment effects. *The Stata Journal* 16(1): 52-71. [10.1177/1536867X1601600108](https://doi.org/10.1177/1536867X1601600108)
  - von Bismarck-Osten C, Borusyak K, Schönberg U (2022) The role of schools in transmission of the SARS-CoV-2 virus: quasi-experimental evidence from Germany. *Economic Policy* 37(109): 87–130. [10.1093/epolic/eiac001](https://doi.org/10.1093/epolic/eiac001)
  - Wieland T (2025) Assessing the effectiveness of non-pharmaceutical interventions in the SARS-CoV-2 pandemic: results of a natural experiment regarding Baden-Württemberg (Germany) and Switzerland in the second infection wave. *Journal of Public Health: From Theory to Practice* 33(11): 2497-2511. [10.1007/s10389-024-02218-x](https://doi.org/10.1007/s10389-024-02218-x)
  - Wooldridge JM (2012) *Introductory Econometrics. A Modern Approach*.


## AI Usage Statement

This software was developed without the use of AI-generated code. The Continue Agent in Microsoft Visual Studio Code using the GPT-5 mini model (by OpenAI) was used solely to assist in drafting and refining docstrings for documentation. The corresponding guidelines and constraints defined by the author are documented in `AGENTS-docstrings.md` in the [public GitHub repository](https://github.com/geowieland/diffindiff_official).


## What's new (v2.3.4)

- Bugfixes:
  - Fixed bug in DiffData instance creation in diddata.merge_data()