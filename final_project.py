import pandas as pd
import numpy as np
import matplotlib as plt

#Initial data exploration
original_df = pd.read_csv('master.csv')
suicide_df = original_df.copy()

suicide_df.columns = ['country', 'year', 'gender', 'age', 'suicides_no', 'population', 'suicides_100k_pop', 'country_year', 'hdi_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation']
suicide_df.drop('country_year', axis=1, inplace=True)
suicide_df.drop('hdi_for_year', axis=1, inplace=True)
suicide_df.drop('generation', axis=1, inplace=True)

suicide_df.loc[suicide_df['country'] == 'United States', 'country'] = 'United States of America'
suicide_df.loc[suicide_df['age'] == '5-14 years', 'age'] = '05-14 years'

#===============================================

#Manipulating null values
suicide_df = pd.read_csv('fixed_values.csv') #See the colab file for the wall of text
print(suicide_df.info())

#========================================================================

#Data Visualization

#1. Total rates analysis
#2. Gender analysis
#3. Age group analysis
#4. Correlation