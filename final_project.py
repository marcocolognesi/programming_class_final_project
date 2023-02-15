import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

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
suicide_df = pd.read_csv('fixed_values.csv') #See the colab file for the wall of text of the manipulation
suicide_df.drop('Unnamed: 0',axis=1, inplace=True)
#========================================================================

#Streamlit presentation and Data visualization
st.title("EDA and Visualization of suicide rates from 1986 to 2021")
st.header('Introduction')
st.write('The aim of this project is to.')
st.write('A look at the raw data:')
st.write(original_df)


st.header('Data Visualization')

#1. Total rates analysis
st.subheader('1. Total Rates Analysis')
year_groupby_suicidesum =suicide_df.groupby('year').suicides_no.sum()
f, ax = plt.subplots(figsize=(16,8))
sns.barplot(x=year_groupby_suicidesum.values , y= year_groupby_suicidesum.index, color = 'c', orient='h')
ax.set_title('Total suicide rates from 1986 to 2020', weight='bold')
#ax.ticklabel_format(axis = 'x', style = 'plain')
ax.set_ylabel('Year')
ax.set_xlabel('Total suicide rates')
st.write(f)

print(suicide_df.info())


#========================================================================
#Data Visualization 

#2. Gender analysis
#3. Age group analysis
#4. Correlation