import pandas as pd
import numpy as np
import matplotlib as plt

#Initial data exploration
original_df = pd.read_csv('master.csv')
suicide_df = original_df.copy()

suicide_df.columns = ['country', 'year', 'gender', 'age', 'suicides_no', 'population', 'suicides_100k_pop', 'country_year', 'hdi_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation']
