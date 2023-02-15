import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

st.title("EDA and Visualization of global suicide rates from 1986 to 2021")
st.markdown(
'''
Suicide is a serious global public health issue. It is among the top twenty leading causes of death worldwide, with more deaths due to suicide than to malaria, breast cancer, or war and homicide. 
Close to 800 000 people die by suicide every year (*World Health Organization, 2019*).
\n 
The aim of this project is to analyze and visualize the different suicide rates around the world, from 1986 to 2021.
'''
)

#=======================================================================================================================================================================================================
#INITIAL DATA EXPLORATION

st.header('Initial data exploration')
st.write('By taking a first look at the dataset, we can see that the rates are divided for each country, for each year, for each sex and for 6 different age groups.')

original_df = pd.read_csv('master.csv')

if st.checkbox('Click to see the original data'):
    st.write(original_df)
    st.caption('The original data can be downloaded from the sidebar.')

#Adding download button and link in the sidebar
st.sidebar.subheader('Useful links')
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
4. We rename the 5-14 age group for a better future sorting and visualization. 
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
original_df = pd.read_csv('master.csv')
suicide_df = original_df.copy()

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
if st.checkbox('Click to see the code', key='1'):
    st.code(first_changes_code, language='python')

#======================================================================================================================================================================================================

#MANIPULATING THE NULL VALUES

st.header('Manipulation of the null values')
st.markdown(
'''
Looking at the info of our dataset, we see that there are some **null values** in the suicide rates column (*precisely 1200 null values*).
\n We can now procede in two ways: either **drop** them or **fix** them.
\n Suicide rates are not an easily predictable number as many factors could influence them. In a certain year there might be a huge increase (*for example due to a big shock in the world economy*) or decrease 
(*for example because of a relatively stable situation*). The best decision should be to drop the nulls, however, although it may result inaccurate, for the sake of this project we decide to manipulate the values 
and replace the null ones with their relative mean from the last 10 years (*mean based on the rates of the last 10 years for each country, age group and gender*).

'''
)

null_example_code = (
'''
#Creating boolean masks
gender_mask = df['gender'] == 'gender' #gender can be 'male' or 'female'
age_mask = df['age'] == 'xx-xx years' #the different age groups are 05-14, 25-34, 35-54, 55-74, 75+
country_mask = df['country'] == 'country' #For example Italy, Japan, Germany, ...

#Replacing example
df.loc[(country_mask) & (age_mask) & (gender_mask), 'suicides_no'] = df[(country_mask) & (age_mask) & (gender_mask)].fillna(np.mean(df[(country_mask) & (age_mask) & (gender_mask) & (df['year'] >= 2010) & (df['suicides_no'] >= 0)]['suicides_no']).round(decimals=0))
'''
)
if st.checkbox('Click to see the code', key='2'):
    st.code(null_example_code, language='python')
    st.caption('This process has been done for each country, each gender and each age group. The whole code can be found in the Google Colab file linked in the Github page.')

st.markdown(
'''
We have now succesfully created a new dataframe without null values. This move, however, messes up the dataframe a little bit. The suicide rates over 100k population related to the previously null values are still
equal to 0. Fixing them is not difficult, however, from 2016 to 2021 the population is sometimes not counted properly like in the previous years (*it uses the population of the whole country for each rate instead of
using the population of each group and gender*).
\n This incongruence is due to the fact that from that year, the data has been update by a different author.
\n For this reason we are gonna use the new dataset with the estimated rates only in certain plots and a dataset until 2016 for others. 
'''
)

suicide_df_filled = pd.read_csv('fixed_values.csv') #See the colab file for the wall of text of the manipulation
suicide_df_filled.drop('Unnamed: 0',axis=1, inplace=True)

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