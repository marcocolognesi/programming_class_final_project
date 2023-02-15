import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

st.title("Data Analysis and Visualization of suicide rates from 1986 to 2021")
st.write('Suicide is a serious global public health issue. It is among the top twenty leading causes of death worldwide, with more deaths due to suicide than to malaria, breast cancer, or war and homicide. Close to 800 000 people die by suicide every year (*World Health Organization, 2019*).')
st.write('The aim of this project is to analyze and visualize the different suicide rates among most of the countries in the world, from 1986 to 2021.')

#=======================================================================================================================================================================================================
#INITIAL DATA EXPLORATION

st.header('Initial data exploration')
st.write('By taking a first look at the dataset, we can see that the rates are divided for each country, for each year, for each sex and for 6 different age groups.')

original_df = pd.read_csv('master.csv')

if st.checkbox('Click to see the original data'):
    st.write(original_df)
    st.caption('The original data can be downloaded from the sidebar.')

#Adding download button and link in the sidebar
url = 'https://www.kaggle.com/datasets/omkargowda/suicide-rates-overview-1985-to-2021'
st.sidebar.write('Dataset source: [click link](' + url + ')')
st.sidebar.download_button('Download original dataset', original_df.to_csv(), file_name='suicide_rates.csv')

st.write('The **libraries** used in this project are:')
importing_code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

'''
st.code(importing_code, language='python')

suicide_df = original_df.copy()

st.write('The changes made to the original data are the following:')
st.markdown(
'''
1. The columns are renamed in order to be more manageable;
2. The following columns are dropped:
    - **'country_year'**, as it's useless for our analysis;
    - **'hdi_for_year'**, as it doesn't have enough data to work with (*only 12.300 values instead of the 31.756 of the other columns*);
    - **'generation'**, because by taking a deeper look into the data we can see that it's a bit inaccurate.
3. We fix the rates for the United States of America, as they are listed in two different ways;
4. We rename the 5-14 age group for a better visualization. 
'''
)

#Renaming columns
suicide_df.columns = ['country', 'year', 'gender', 'age', 'suicides_no', 'population', 'suicides_100k_pop', 'country_year', 'hdi_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation']

#Dropping useless columns
suicide_df.drop('country_year', axis=1, inplace=True) #useless
suicide_df.drop('hdi_for_year', axis=1, inplace=True) #not enough data
suicide_df.drop('generation', axis=1, inplace=True) #inaccurate

#Fixing USA values and renaming the 05-14 age group
suicide_df.loc[suicide_df['country'] == 'United States', 'country'] = 'United States of America'
suicide_df.loc[suicide_df['age'] == '5-14 years', 'age'] = '05-14 years'

first_changes_code = (
'''
#Renaming columns
suicide_df.columns = ['country', 'year', 'gender', 'age', 'suicides_no', 'population', 'suicides_100k_pop', 'country_year', 'hdi_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation']

#Dropping useless columns
suicide_df.drop('country_year', axis=1, inplace=True)
suicide_df.drop('hdi_for_year', axis=1, inplace=True)
suicide_df.drop('generation', axis=1, inplace=True)

#Fixing USA values and renaming the 05-14 age group
suicide_df.loc[suicide_df['country'] == 'United States', 'country'] = 'United States of America'
suicide_df.loc[suicide_df['age'] == '5-14 years', 'age'] = '05-14 years'
'''
)
st.code(first_changes_code, language='python')
#======================================================================================================================================================================================================

#MANIPULATING THE NULL VALUES

suicide_df = pd.read_csv('fixed_values.csv') #See the colab file for the wall of text of the manipulation
suicide_df.drop('Unnamed: 0',axis=1, inplace=True)

#========================================================================

#Streamlit presentation and Data visualization
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