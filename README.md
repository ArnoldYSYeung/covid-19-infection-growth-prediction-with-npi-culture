# COVID-19 Confirmed Infection Growth Prediction with Non-Pharmaceutical Interventions and Cultural Dimensions

The released publication of this work may be found here: 
- <a href='https://www.jmir.org/2021/4/e26628'>**Machine Learning–Based Prediction of Growth in Confirmed COVID-19 Infection Cases in 114 Countries Using Metrics of Nonpharmaceutical Interventions and Cultural Dimensions: Model Development and Validation**</a> (Journal of Medical Internet Research, 2021).

The main pipeline used in this study is in `Experiment.ipynb`. The library `ml_pipeline.py` contains functions used in the pipeline.

There are two directories:
- `./data/` contains the 3 data sets used in this study
- `./figures/` contains the figures shown in the publication. Running `Experiment.ipynb` will generate figures in this directory



## Run
Running `Experiment.ipynb` will generate the tables and figures shown in the publication. Settings of the study may be modified in cell 4 of this notebook. These settings include:
- `time_series_split_method`: `True` runs the out-of-distribution validation method, `False` runs the in-distribution validation method. The country-based cross-validation method is ran additionally, regardless of the value of this parameter. 
- `run_models`: Boolean dictionary indicating which models to run. `True` includes the model in the experiment. `False` excludes the model from the experiment.
- `times_new_roman`: `True` generates figures with the Times New Roman font. `False` generates figures with the default font.

## Acknowledgements
This work is co-authored by Arnold YS Yeung, Francois Roewer-Despres, Laura Rosella, and Frank Rudzicz.
