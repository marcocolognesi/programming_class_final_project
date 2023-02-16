import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

st.set_page_config(layout='centered')
st.title("EDA and Visualization of global suicide rates from 1985 to 2020")
st.markdown(
'''
Suicide is a serious global public health issue. It is among the top twenty leading causes of death worldwide, with more deaths due to suicide than to malaria, breast cancer, or war and homicide. 
Close to 800 000 people die by suicide every year (*World Health Organization, 2019*).
\n 
The aim of this project is to analyze and visualize the different suicide rates around the world, from 1985 to 2020.
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

st.write('The **libraries** used in this project are: **pandas**, **numpy**, **matplotlib**, **seaborn** and **streamlit**')
importing_code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

'''
if st.checkbox('Click to see the code', key='1'):
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
if st.checkbox('Click to see the code', key='2'):
    st.code(first_changes_code, language='python')

#======================================================================================================================================================================================================

#MANIPULATING THE NULL VALUES

st.header('Manipulation of the null values')
st.markdown(
'''
Looking at the info of our dataset, we see that there are some **null values** in the suicide rates column (*precisely 1200 null values, which are only from 2017 to 2020*).
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
country_mask = df['country'] == 'xyz' #For example Italy, Japan, Germany, ...

#Replacing example
df.loc[(country_mask) & (age_mask) & (gender_mask), 'suicides_no'] = df[(country_mask) & (age_mask) & (gender_mask)].fillna(np.mean(df[(country_mask) & (age_mask) & (gender_mask) & (df['year'] >= 2010) & (df['suicides_no'] >= 0)]['suicides_no']).round(decimals=0))
'''
)
if st.checkbox('Click to see the code', key='3'):
    st.code(null_example_code, language='python')
    st.caption('This process has been done for each country, each gender and each age group. The whole code can be found in the Google Colab file linked in the Github page.')

st.markdown(
'''
We have now succesfully created a new dataframe without null values. This move, however, messes up the dataframe a little bit. The suicide rates over 100k population related to the previously null values are still
equal to 0. Fixing them is not difficult, however, from 2016 to 2020 the population is sometimes not counted properly like in the previous years (*it uses the population of the whole country for each rate instead of
using the population of each group and gender*).
\n This incongruence is due to the fact that from that year, the data has been update by a different author.
\n For this reason we are gonna use the new dataset with the estimated rates only in certain plots and a dataset until 2016 for others. 
'''
)

#Creating the new datasets
suicide_df_filled = pd.read_csv('fixed_values.csv') #See the colab file for the wall of text of the manipulation
suicide_df_filled.drop('Unnamed: 0',axis=1, inplace=True)

suicide_df_not_filled = suicide_df.dropna()

year_2016_mask = suicide_df_not_filled['year'] < 2016
suicide_df_until_2016 = suicide_df_not_filled[year_2016_mask]

#==========================================================================================================================================================================================================

#DATA VISUALIZATION

st.header('Data Visualization')

#1. Total rates analysis
st.subheader('Analysis of the total rates from 1985 to 2020')

year_groupby_sumrates_filled = suicide_df_filled.groupby('year').suicides_no.sum()

option = st.selectbox(
    'What type of plot you want to display?',
    ('Choose the plot','Bar plot', 'Line plot'))
if option == 'Bar plot':
    f, ax = plt.subplots(figsize=(12,10))
    sns.barplot(x=year_groupby_sumrates_filled.values , y= year_groupby_sumrates_filled.index, color = 'c', orient='h')
    ax.set_title('Trend of the total suicide rates from 1985 to 2020', weight='bold')
    #ax.ticklabel_format(axis = 'x', style = 'plain')
    ax.set_ylabel('Year')
    ax.set_xlabel('Total suicide rates')
    st.write(f)
elif option == 'Line plot':
    f, ax = plt.subplots(figsize=(12,10))
    sns.lineplot(x=year_groupby_sumrates_filled.index , y= year_groupby_sumrates_filled.values)
    ax.set_title('Trend of the total suicide rates from 1985 to 2020', weight='bold')
    #ax.ticklabel_format(axis = 'x', style = 'plain')
    ax.set_ylabel('Total suicide rates')
    ax.set_xlabel('Year')
    st.write(f)
else:
    None


st.caption('Trend of the total suicide rates from 1985 to 2020 (*using the data with the estimated values obtained with the mean*)')
st.markdown(
'''
Looking at the trend of the total suicide rates from 1985 to 2020, we can see that the peak has occurred in the late 1990's - early 2000's, in particular from 1999 to 2003.
\n Since then, the trend went down and reached numbers similar to the ones from the early 1990's. Also, the huge decrease in 2016 is due to the fact that we have a lot less data for that year (*only 160 values*).
'''
)

# 2. Top 10 countries analysis

st.subheader('Countries with most rates analysis')
st.write(
'''
Let's see which are the top 10 countries with the most rates over the years, comparing also by gender and age.
'''
)

#Finding out which are the top 10 countries with the most suicide rates over the years
country_groupby_sumrates_not_filled = suicide_df_not_filled.groupby('country').suicides_no.sum()
top_10_countries = country_groupby_sumrates_not_filled.sort_values(ascending=False)[0:10].index
top_10_most_suicide_rates = country_groupby_sumrates_not_filled.sort_values(ascending=False)[0:10].values

state_list = ['Russian Federation', 'United States of America', 'Japan', 'Ukraine', 'France','Germany','Republic of Korea','Brazil', 'Poland', 'United Kingdom']
labels = ['Russia', 'USA', 'Japan', 'Ukraine', 'France', 'Germany', 'South Korea', 'Brazil', 'Poland', 'UK']

top_10_countries_most_suicides_rates_df_not_filled = suicide_df_not_filled.query('country in @state_list')

top_10_countries_gender_groupby_sumrates_not_filled = top_10_countries_most_suicides_rates_df_not_filled.groupby(['country','gender']).suicides_no.sum()
top_10_countries_gender_groupby_sumrates_not_filled_sorted = top_10_countries_gender_groupby_sumrates_not_filled.reset_index().set_index('country').loc[state_list]

top_10_countries_age_groupby_sumrates_not_filled = top_10_countries_most_suicides_rates_df_not_filled.groupby(['country','age']).suicides_no.sum()
top_10_countries_age_groupby_sumrates_not_filled_sorted = top_10_countries_age_groupby_sumrates_not_filled.reset_index().set_index('country').loc[state_list]

#Plotting results
genre = st.radio(
    'Choose the plot you want to see',
    ('Total rates comparison', 'Gender comparison', 'Age groups comparison')
)
if genre == 'Total rates comparison':
    f, ax = plt.subplots(figsize=(12,10))
    sns.barplot(x= top_10_countries, y= top_10_most_suicide_rates, palette = "rocket")
    ax.set_title('Top 10 countries with most suicides from 1986 to 2020', weight='bold')
    ax.ticklabel_format(axis = 'y', style = 'plain')
    ax.set_ylabel('Total rates')
    ax.set_xlabel('Countries')
    ax.set_xticklabels(labels)
    st.write(f)
elif genre == 'Gender comparison':
    f, ax = plt.subplots(figsize=(12,10))
    sns.barplot(x= top_10_countries_gender_groupby_sumrates_not_filled_sorted.index, y= 'suicides_no', hue='gender', data= top_10_countries_gender_groupby_sumrates_not_filled_sorted, palette=('hotpink','cornflowerblue'))
    ax.set_title('Top 10 countries with most suicides from 1986 to 2020, gender comparison', weight='bold')
    ax.ticklabel_format(axis = 'y', style = 'plain')
    ax.set_ylabel('Total rates')
    ax.set_xlabel('Gender')
    ax.set_xticklabels(labels)
    st.write(f)
else:
    f, ax = plt.subplots(figsize=(12,10))
    sns.barplot(x= top_10_countries_age_groupby_sumrates_not_filled_sorted.index, y= 'suicides_no', hue='age', data= top_10_countries_age_groupby_sumrates_not_filled_sorted, palette='rocket_r')
    ax.set_title('Top 10 countries with most suicides from 1986 to 2020, age comparison', weight='bold')
    ax.ticklabel_format(axis = 'y', style = 'plain')
    ax.set_ylabel('Total rates')
    ax.set_xlabel('Age groups')
    ax.set_xticklabels(labels)
    st.write(f)

st.caption('Top 10 countries with the most suicide rates over from 1985 to 2020, using the real data we have in this set (***not using the estimated values obtained with the mean***)')
st.markdown(
'''
Looking at the plot, we see that the most suicides occurred in Russia and USA. This shouldn't be surprising as these countries are among the largest in the world and with more population than the others.
Also, we need to take into consideration the fact that we don't have reliable data (*or no data at all*) for a lot of important countries, such as China and India for example, but also for most of the african ones.
- The **gender comparison** shows us that among these countries we had **more male suicides than female ones**. 
- Japan seems to be the country with most female suicides.
- The **age group comparison** shows us that in each of the countries in this list, **most of the rates occurred in the 35-54 age group**. From this comparison we see also that Japan has the most rates within the 15-24 and 55-74 age group. We can also see that Brazil has a strangely high rate in the 15-24 age group, in comparison with the other countries near his position.
'''
)

#2. Gender analysis
#3. Age group analysis
#4. Correlation