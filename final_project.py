import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

#Web page: Title and info box

st.set_page_config(layout='centered')

st.title("EDA and Visualization of global suicide rates from 1985 to 2020")

st.info(
'''
**Suicide is a serious global public health issue**. It is among the **top twenty leading causes of death worldwide**, with more deaths due to suicide than to malaria, breast cancer, or war and homicide. 
Close to 800 000 people die by suicide every year (***World Health Organization, 2019***).
\n 
The aim of this project is to **analyze** and **visualize** the different suicide rates around the world and how they changed over the years (***from 1985 to 2020***), and also to see the **correlation** between our data. In particular, we are interested to know if the GDP per capita is a factor that influences the rates.
\n In the last steps of our analysis we are gonna implement some basic **Machine Learning** examples such as **clustering** and **linear regression**.
'''
)

#=======================================================================================================================================================================================================
# 1. INITIAL DATA EXPLORATION

# 1st Chapter header
st.header('Initial data exploration')

#creating original df and a copy to work with
original_df = pd.read_csv('master.csv')
suicide_df = original_df.copy()

#adding text and checkbox on streamlit
st.write('By taking a first look at the dataset, we can see how the rates are divided for each **country**, for each **year** (*from 1985 to 2020*), for each **sex** and for 6 different **age groups** (*"05-14 years", "15-24 years", "25-34 years", "35-54 years", "55-74 years", "75+ years"*).')

#checkbox to see the original data in the web page
if st.checkbox('Click to see the original data'):
    st.write(original_df)
    st.caption('The original data can be downloaded from the sidebar.')

#===============================================================================================================
# SIDEBAR
#Adding download button and link in the sidebar
st.sidebar.subheader('Useful links')

url = 'https://www.kaggle.com/datasets/omkargowda/suicide-rates-overview-1985-to-2021'

st.sidebar.write('Dataset source: [click link](' + url + ')')

st.sidebar.download_button('Download original dataset', original_df.to_csv(), file_name='suicide_rates.csv')
#===============================================================================================================

#Text, importing_code_example, list of changes made + code examples

st.write('The **libraries** used in this project are: **pandas**, **numpy**, **matplotlib**, **seaborn**, **sklearn** and **streamlit**')

importing_code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

#Machine Learning part
#Clustering
from sklearn.cluster import KMeans

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
'''

if st.checkbox('Click to see the code', key='1'):
    st.code(importing_code, language='python')

st.write('''
The **changes** made to the original data are the following:

* The columns are **renamed** in order to be more manageable;
* The following columns are **dropped**:
    - **'country_year'**, because it's useless for our analysis;
    - **'hdi_for_year'**, because it doesn't have enough data to work with (*only 12.300 values instead of the 31.756 of the other columns*);
    - **'generation'**, because by taking a deeper look into the data we can see that it's a bit inaccurate.
* We fix the rates for the United States of America, as they are listed in two different ways;
* We rename the "5-14 years" age group for a better future sorting and visualization. 
'''
)

#=================================================================================================================================
#CHANGES IN THE DATASET
# 1. Renaming columns
suicide_df.columns = ['country', 'year', 'gender', 'age', 'suicides_no', 'population', 'suicides_100k_pop', 'country_year', 'hdi_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation']

# 2. Dropping useless columns
suicide_df.drop('country_year', axis=1, inplace=True) #useless
suicide_df.drop('hdi_for_year', axis=1, inplace=True) #not enough data
suicide_df.drop('generation', axis=1, inplace=True) #inaccurate

# 3. Fixing USA values and renaming the 05-14 age group
suicide_df.loc[suicide_df['country'] == 'United States', 'country'] = 'United States of America'
suicide_df.loc[suicide_df['age'] == '5-14 years', 'age'] = '05-14 years'
#=================================================================================================================================

#Implementing code example in streamlit

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
st.write(
'''
Looking at the info of our dataset, we see that there are some **null values** in the suicide rates column (***precisely 1200 null values, which are only from 2017 to 2020***).
\n We can now procede in two ways: either **drop** them or **fix** them.
\n Suicide rates are not an easily predictable number as many factors may influence them. In a certain year there might be a huge increase (***for example due to a big shock in the world economy***) or decrease 
(***for example because of a relatively stable situation***). The best decision should be to **drop the nulls**, however, although it may result inaccurate, for the sake of this project we decide to manipulate the values 
and **replace** the null ones with their relative **mean from the last 10 years** (***mean based on the rates of the last 10 years for each country, age group and gender***).

'''
)

#=============================================================================
#Code example
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
#=============================================================================

#Continuing text

st.write(
'''
We have now succesfully created a **new dataframe without null values**. This move, however, messes up the dataframe a little bit. The suicide rates over 100k population related to the previously null values are still
equal to 0. Fixing them is not difficult, however, from 2016 to 2020 the population is sometimes not counted properly like in the previous years (***it uses the population of the whole country for each rate instead of
using the population of each group and gender***).
\n This incongruence is due to the fact that from that year, the data has been update by a **different author**.
'''
)

st.error('''
**For this reason we are gonna use the new dataset with the estimated rates only in certain plots and a dataset with the values until 2016 for others**.

'''
)

#Creating the new datasets
suicide_df_filled = pd.read_csv('fixed_values.csv') #See the colab file for the wall of text of the manipulation
suicide_df_filled.drop('Unnamed: 0',axis=1, inplace=True)

suicide_df_not_filled = suicide_df.dropna()

year_2016_mask = suicide_df_not_filled['year'] < 2016
suicide_df_until_2016 = suicide_df_not_filled[year_2016_mask]

#==========================================================================================================================================================================================================

#DATA VISUALIZATION CHAPTER

st.header('Data Visualization')

#1. Total rates analysis
st.subheader('Analysis of the total rates from 1985 to 2020')

#Year groupby, using the dataframe with estimated values
year_groupby_sumrates_filled = suicide_df_filled.groupby('year').suicides_no.sum()
year_groupby_sumrates_filled = year_groupby_sumrates_filled.reset_index()

#Total Rates plot
option = st.selectbox(
    'What type of plot you want to display?',
    ('Choose the plot','Bar plot', 'Line plot'),
    key = 'selectbox1'
)
if option == 'Bar plot':
    #Bar plot
    f, ax = plt.subplots(figsize=(12,10))

    ax.set_title('Trend of the total suicide rates from 1985 to 2020', weight='bold')

    sns.barplot(x='year' , y= 'suicides_no', data=year_groupby_sumrates_filled, color = 'c', orient='v')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Total suicide rates')
    ax.tick_params(axis='x', labelrotation = 45)
    
    st.write(f)
    st.caption('Trend of the total suicide rates from 1985 to 2020 (***using the data with the estimated values obtained with the mean***)')
elif option == 'Line plot':
    #Line plot
    f, ax = plt.subplots(figsize=(12,10))

    ax.set_title('Trend of the total suicide rates from 1985 to 2020', weight='bold')

    sns.lineplot(x='year' , y= 'suicides_no', data=year_groupby_sumrates_filled)
    
    ax.set_ylabel('Total suicide rates')
    ax.set_xlabel('Year')
    
    st.write(f)
    st.caption('Trend of the total suicide rates from 1985 to 2020 (***using the data with the estimated values obtained with the mean***)')
else:
    st.text('')

#Writing conclusions about the total rates over the years analysis
st.write(
'''
Looking at the trend of the **total suicide rates from 1985 to 2020**, we can see that the **peak** has occurred in the **late 1990's - early 2000's**, in particular from **1999** to **2003**.
\n Since then, the trend went down and reached numbers similar to the ones from the early 1990's. Also, **the huge decrease in 2016 is due to the fact that we have a lot less data for that year** (*only 160 values*).
\n Now, let's see how the total rates are divided by **sex** and **age group** and also how they changed over the **years**:
'''
)


#GENDER AND AGE TOTAL RATES PLOTS

#GENDER GROUPBY
#total rates
gender_groupby_sumrates_not_filled = suicide_df_not_filled.groupby('gender').suicides_no.sum()
gender = gender_groupby_sumrates_not_filled.index
values = gender_groupby_sumrates_not_filled.values
year_gender_groupby_not_filled = suicide_df_not_filled.groupby(['year','gender'])
year_gender_groupby_sumrates_not_filled = year_gender_groupby_not_filled.suicides_no.sum().reset_index()
#total rates over 100k
gender_groupby_sumrates_100kpop_not_filled = suicide_df_not_filled.groupby('gender').suicides_100k_pop.sum()
gender_100k = gender_groupby_sumrates_100kpop_not_filled.index
values_100k = gender_groupby_sumrates_100kpop_not_filled.values
year_gender_groupby_sumrates_100kpop_not_filled = year_gender_groupby_not_filled.suicides_100k_pop.sum().reset_index()

#AGE GROUPBY
#total rates
age_groupby_sumrates_not_filled = suicide_df_not_filled.groupby('age').suicides_no.sum()
rocket_palette = sns.color_palette('rocket_r')
year_age_groupby_not_filled = suicide_df_not_filled.groupby(['year','age'])
year_age_groupby_sumrates_not_filled = year_age_groupby_not_filled.suicides_no.sum().reset_index()
#total rates over 100k
age_groupby_sumrates_100kpop_not_filled = suicide_df_not_filled.groupby('age').suicides_100k_pop.sum()
year_age_groupby_sumrates_100kpop_not_filled = year_age_groupby_not_filled.suicides_100k_pop.sum().reset_index()


#PLOTTING RESULTS
option2 = st.selectbox(
    'What plot you want to display?',
    ('Choose the plot','Gender comparison', 'Age comparison'),
    key = 'selectbox2'
)
if option2 == 'Gender comparison':
    st.write('***Comparison of the total rates over the years, by gender***:')
    
    col_1,col_2 = st.columns(2)
    with col_1:
        f, ax = plt.subplots(1,2, figsize = (12,10))

        f.suptitle('Comparison of male and female total suicide rates from 1985 to 2020', weight='bold')
        
        sns.barplot(x= gender, y= values, palette =('hotpink','cornflowerblue'), ax=ax[1])
        
        ax[1].ticklabel_format(axis = 'y', style = 'plain')
        ax[1].set_ylabel('Total suicide rates')
        ax[1].set_xlabel('Sex')
        
        gender_groupby_sumrates_not_filled.plot.pie(colors=('hotpink','cornflowerblue'), explode=[0, 0.01], autopct='%1.1f%%', ax=ax[0])
        ax[0].set_ylabel('')
        
        st.write(f)
    with col_2:
        f, ax = plt.subplots(figsize = (12,10.709))
        
        f.suptitle('Comparison of male and female total suicide rates from 1985 to 2020 for each year', weight='bold')

        sns.barplot(y = 'suicides_no', x = 'year', hue = 'gender',data = year_gender_groupby_sumrates_not_filled, palette =('hotpink','cornflowerblue'), orient='v')
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total suicide rates')
        
        ax.tick_params(axis='x', labelrotation = 45)
        st.write(f)

    st.caption('The left plot shows the comparison of male and female total suicide rates from 1985 to 2020. The right plot, instead, shows the total male and female suicide rates for each year (***both plots are obtained using the data without the estimated values***)')
    
    st.write('***Comparison of the total rates over 100k population, over the years and by gender***:')
    
    col_1, col_2 = st.columns(2)
    with col_1:
        f, ax = plt.subplots(1,2, figsize = (12,10))
        f.suptitle('Comparison of male and female total suicide rates over 100k population from 1985 to 2020', weight='bold')
        
        sns.barplot(x= gender_100k, y= values_100k, palette =('hotpink','cornflowerblue'), ax=ax[1])
        
        ax[1].ticklabel_format(axis = 'y', style = 'plain')
        ax[1].set_xlabel('Sex')
        ax[1].set_ylabel('Total suicide rates over 100k population')
        
        gender_groupby_sumrates_100kpop_not_filled.plot.pie(colors=('hotpink','cornflowerblue'), explode=[0, 0.01], autopct='%1.1f%%', ax=ax[0])
        
        ax[0].set_ylabel('')
        
        st.write(f)    
    with col_2:
        f, ax = plt.subplots(figsize = (12,10.789))
        
        f.suptitle('Comparison of male and female total suicide rates over 100k population from 1985 to 2020, for each year', weight='bold')
        
        sns.barplot(y = 'suicides_100k_pop', x = 'year', hue = 'gender',data = year_gender_groupby_sumrates_100kpop_not_filled, palette =('hotpink','cornflowerblue'), orient='v')
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total suicide rates over 100k population')
        ax.tick_params(axis='x', labelrotation = 45)
        
        st.write(f)    
    st.caption('The left plot shows the comparison of **male and female total suicide rates over 100k population** from 1985 to 2020. The right plot, instead, shows the total male and female suicide rates over 100k population for each year (***both plots are obtained using the data without the estimated values***)')
if option2 == 'Age comparison':

    st.write('***Comparison of the total rates over the years, by different age groups***:')
    
    col_1, col_2 = st.columns(2)
    with col_1:
        f, ax = plt.subplots(1,2, figsize=(12,10))
        
        f.suptitle('Total suicide rates comparison for each age group from 1985 to 2020', weight='bold')
        
        age_groupby_sumrates_not_filled.plot.pie(ax=ax[0], autopct='%1.1f%%', colors = rocket_palette, labeldistance=1.02, startangle=180, counterclock=False, textprops={'color':"w"})
        
        ax[0].set_ylabel('')
        ax[0].legend(loc='upper left')
        
        sns.barplot(x= age_groupby_sumrates_not_filled.index, y= age_groupby_sumrates_not_filled.values, palette = 'rocket_r', ax=ax[1])
        
        ax[1].tick_params(axis='x', labelrotation = 45)
        ax[1].ticklabel_format(axis = 'y', style = 'plain')
        ax[1].set_ylabel('Total suicide rates')
        
        ax[1].set_xlabel('Age group')
        st.write(f)
    with col_2:
        f, ax = plt.subplots(figsize=(12,10.9))
        
        f.suptitle('Comparison of the total suicide rates for each year and age group from 1985 to 2020', weight='bold')
        
        sns.barplot(x = 'year', y = 'suicides_no', hue = 'age',data = year_age_groupby_sumrates_not_filled, palette = 'rocket_r', orient='v')
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total suicide rates')
        ax.tick_params(axis='x', labelrotation = 45)
        
        st.write(f)
    st.caption('The left plot shows the comparison of the **different age groups total suicide rates** from 1985 to 2020. The right plot, instead, shows the total suicide rates among the different age groups for each year (***both plots are obtained using the data without the estimated values***)')
    
    st.write('***Comparison of the total rates over 100k population, over the years and by different age groups***:')
    
    col_1, col_2 = st.columns(2)
    with col_1:
        f, ax = plt.subplots(1,2, figsize = (12,10))
        
        f.suptitle('Total suicide rates over 100k population comparison for each age group from 1985 to 2020', weight='bold')
        
        sns.barplot(x= age_groupby_sumrates_100kpop_not_filled.index, y= age_groupby_sumrates_100kpop_not_filled.values, palette = 'rocket_r', ax=ax[1])
        
        ax[1].ticklabel_format(axis = 'y', style = 'plain')
        ax[1].set_ylabel('Total suicide rates over 100k population')
        ax[1].set_xlabel('Age group')
        ax[1].tick_params(axis='x', labelrotation = 45)
        
        age_groupby_sumrates_100kpop_not_filled.plot.pie(ax=ax[0], autopct='%1.1f%%', colors = rocket_palette, labeldistance=1.02, startangle=180, counterclock=False, textprops={'color':"w"})
        
        ax[0].set_ylabel('')
        ax[0].legend(loc='upper left')
        
        st.write(f)
    with col_2:
        f, ax = plt.subplots(figsize=(12,10.89))
        
        f.suptitle('Comparison of the total suicide rates over 100k population for each year and age group from 1985 to 2020', weight='bold')
        
        sns.barplot(x = 'year', y = 'suicides_100k_pop', hue = 'age',data = year_age_groupby_sumrates_100kpop_not_filled, palette = 'rocket_r', orient='v')
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total suicide rates over 100k population')
        ax.tick_params(axis='x', labelrotation = 45)
        
        st.write(f)
    st.caption('The left plot shows the comparison of the **different age groups total suicide rates over 100k population** from 1985 to 2020. The right plot, instead, shows the total suicide rates among the different age groups over 100k population and for each year (***both plots are obtained using the data without the estimated values***)')
else:
    st.write('')

st.write(
'''
* From the **gender analysis**, we can see that, over the years, the **number of male suicides** has been **far superior** compared to the **female rates**. Also, the **analysis of the rates over 100k population confirms this statement**.
* From the **age groups analysis**, we can see that, over the years, the rates of the **"35-54 years"** age group has been **far superior** compared to the others. The **analysis of the rates over 100k population**, however, does not confirm this statement. In fact, in this last analysis, the "**75+ years**" age group has the most rates over the years.

Now, let's see which are the countries around the world with the most rates over the years.
'''
)

#=================================================
# 2. TOP 10 COUNTRIES ANALYSIS

st.subheader('Countries with the most rates analysis')
st.write(
'''
The steps of this analysis will be the following:
\n * First, we are gonna find the countries with the most rates overall;
* Then, we're gonna find the countries with the most rates over 100k inhabitants.
\n Lastly, we are gonna compare the results and see if there's some similarity between the two groups we are gonna find.
'''
)

st.write(
'''
##### 1. Countries with most rates analysis
'''
)

#Finding out which are the top 10 countries with the most suicide rates over the years
country_groupby_sumrates_not_filled = suicide_df_not_filled.groupby('country').suicides_no.sum()
top_10_countries = country_groupby_sumrates_not_filled.sort_values(ascending=False)[0:10].index
top_10_most_suicide_rates = country_groupby_sumrates_not_filled.sort_values(ascending=False)[0:10].values

state_list = ['Russian Federation', 'United States of America', 'Japan', 'Ukraine', 'France','Germany','Republic of Korea','Brazil', 'Poland', 'United Kingdom']
labels = ['Russia', 'USA', 'Japan', 'Ukraine', 'France', 'Germany', 'South Korea', 'Brazil', 'Poland', 'UK']

#Creating df of top 10 countries
top_10_countries_most_suicides_rates_df_not_filled = suicide_df_not_filled.query('country in @state_list')

#gender groupby
top_10_countries_gender_groupby_sumrates_not_filled = top_10_countries_most_suicides_rates_df_not_filled.groupby(['country','gender']).suicides_no.sum()
top_10_countries_gender_groupby_sumrates_not_filled_sorted = top_10_countries_gender_groupby_sumrates_not_filled.reset_index().set_index('country').loc[state_list]

#age groupby
top_10_countries_age_groupby_sumrates_not_filled = top_10_countries_most_suicides_rates_df_not_filled.groupby(['country','age']).suicides_no.sum()
top_10_countries_age_groupby_sumrates_not_filled_sorted = top_10_countries_age_groupby_sumrates_not_filled.reset_index().set_index('country').loc[state_list]

with st.expander('Show the plots:'):
    #Plotting results
    genre = st.radio(
        'Choose the **comparison** you want to see',
        ('Total rates', 'Gender', 'Age groups'),
        key='radio1'
    )
    if genre == 'Total rates':
        f, ax = plt.subplots(figsize=(12,10))
        
        ax.set_title('Top 10 countries with most suicides from 1985 to 2020', weight='bold')
        
        sns.barplot(x= top_10_countries, y= top_10_most_suicide_rates, palette = "rocket")
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_ylabel('Total rates')
        ax.set_xlabel('Countries')
        ax.set_xticklabels(labels)
        
        st.write(f)
    elif genre == 'Gender':
        f, ax = plt.subplots(figsize=(12,10))
        
        ax.set_title('Top 10 countries with most suicides from 1985 to 2020, gender comparison', weight='bold')

        sns.barplot(x= top_10_countries_gender_groupby_sumrates_not_filled_sorted.index, y= 'suicides_no', hue='gender', data= top_10_countries_gender_groupby_sumrates_not_filled_sorted, palette=('hotpink','cornflowerblue'))
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_ylabel('Total rates')
        ax.set_xlabel('Countries')
        ax.set_xticklabels(labels)
        
        st.write(f)
    else:
        f, ax = plt.subplots(figsize=(12,10))
        
        ax.set_title('Top 10 countries with most suicides from 1985 to 2020, age comparison', weight='bold')

        sns.barplot(x= top_10_countries_age_groupby_sumrates_not_filled_sorted.index, y= 'suicides_no', hue='age', data= top_10_countries_age_groupby_sumrates_not_filled_sorted, palette='rocket_r')
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_ylabel('Total rates')
        ax.set_xlabel('Countries')
        
        ax.set_xticklabels(labels)
        st.write(f)
    st.caption('Top 10 countries with the most suicide rates over from 1985 to 2020, using the real data we have in this set (***not using the estimated values obtained with the mean***)')

st.write(
'''
Looking at the plot, we see that the **most suicides** occurred in **Russia** and **USA**. This result shouldn't be surprising as these countries are among the largest in the world and with more population than the others.
Also, we need to take into consideration the fact that we don't have reliable data (***or no data at all***) for a lot of important countries, such as China and India for example, but also for most of the african ones.
* The **gender comparison** reflects the result we've seen before from the global trend. In every of this list we had **more male suicides than female ones**:
  - If we look at the female rates, **Japan** seems to be the country with **most female suicides**.
* Also, the **age group comparison** reflects the result we've seen before. In each of the countries in this list, **most of the rates occurred in the 35-54 age group**. From this comparison we see also that:
  - **Japan** has the **most rates within the "15-24 years" and "55-74 years" age group**.
  - **Brazil** has a **strangely high rate in the "15-24 years" age group**, in comparison with the other countries near his position.
'''
)

st.write(
'''
##### 2. Countries with most rates over 100k population analysis
'''
)

#Finding out which are the top 10 countries with the most suicide rates over 100k

country_groupby_sumrates100kpop_not_filled = suicide_df.groupby('country').suicides_100k_pop.sum()
top_10_countries_100kpop = country_groupby_sumrates100kpop_not_filled.sort_values(ascending = False)[0:10].index
top_10_most_suicide_rates_100kpop = country_groupby_sumrates100kpop_not_filled.sort_values(ascending = False)[0:10].values

state_list2 = ['Republic of Korea', 'Russian Federation', 'Lithuania', 'Hungary', 'Kazakhstan', 'Austria', 'Ukraine', 'Japan', 'Finland', 'Belgium']
labels2 = ['South Korea', 'Russia', 'Lithuania', 'Hungary', 'Kazakhstan', 'Austria','Ukraine','Japan','Finland','Belgium']

top_10_countries_most_suicides_rates100kpop_df_not_filled = suicide_df_not_filled.query('country in @state_list2')

#gender groupby
top_10_countries_most_suicides100kpop_gender_groupby_sum_df_not_filled = top_10_countries_most_suicides_rates100kpop_df_not_filled.groupby(['country', 'gender']).suicides_100k_pop.sum()
top_10_countries_most_suicides100kpop_gender_groupby_sum_df_not_filled_sorted = top_10_countries_most_suicides100kpop_gender_groupby_sum_df_not_filled.reset_index().set_index('country').loc[state_list2]

#age groupby
top_10_countries_most_suicides100kpop_age_groupby_sum_df_not_filled = top_10_countries_most_suicides_rates100kpop_df_not_filled.groupby(['country','age']).suicides_100k_pop.sum()
top_10_countries_most_suicides100kpop_age_groupby_sum_df_not_filled_sorted = top_10_countries_most_suicides100kpop_age_groupby_sum_df_not_filled.reset_index().set_index('country').loc[state_list2]

with st.expander('Show the plots'):
    #Plotting results
    genre = st.radio(
        'Choose the **comparison** you want to see',
        ('Total rates', 'Gender', 'Age groups'),
        key='radio2'
    )
    if genre == 'Total rates':
        f, ax = plt.subplots(figsize=(12,10))
        
        ax.set_title('Top 10 countries with most suicides over 100k population from 1986 to 2020', weight='bold')

        sns.barplot(x= top_10_countries_100kpop, y= top_10_most_suicide_rates_100kpop, palette = "rocket")
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_ylabel('Total rates over 100k population')
        ax.set_xlabel('Countries')
        
        ax.set_xticklabels(labels2)
        st.write(f)
    elif genre == 'Gender':
        f, ax = plt.subplots(figsize=(12,10))
        
        ax.set_title('Top 10 countries with most suicides over 100k population from 1986 to 2020, gender comparison', weight='bold')

        sns.barplot(x= top_10_countries_most_suicides100kpop_gender_groupby_sum_df_not_filled_sorted.index, y= 'suicides_100k_pop', hue='gender', data= top_10_countries_most_suicides100kpop_gender_groupby_sum_df_not_filled_sorted, palette=('hotpink','cornflowerblue'))
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_ylabel('Total rates over 100k population')
        ax.set_xlabel('Countries')
        ax.set_xticklabels(labels2)
        
        st.write(f)
    else:
        f, ax = plt.subplots(figsize=(12,10))
    
        ax.set_title('Top 10 countries with most suicides over 100k population from 1986 to 2020, age comparison', weight='bold')

        sns.barplot(x= top_10_countries_most_suicides100kpop_age_groupby_sum_df_not_filled_sorted.index, y= 'suicides_100k_pop', hue='age', data= top_10_countries_most_suicides100kpop_age_groupby_sum_df_not_filled_sorted, palette='rocket_r')
        
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.set_ylabel('Total rates over 100k population')
        ax.set_xlabel('Countries')
        ax.set_xticklabels(labels2)
        
        st.write(f)
    st.caption('Top 10 countries with the most suicide rates over 100k population from 1985 to 2020, using the real data we have in this set (***not using the estimated values obtained with the mean***)')

st.write(
'''
Looking at the plot, we see that the **most suicides over 100k population** occurred in **South Korea**, which was in the 7$^{th}$ position in the analysis we did previously, and **Russia**, which was in first position before.
The fact that Russia is at the top positions in both the two analysis shows that **suicide** is a **huge problem** in that country. Also, a part from Korea, Russia, Ukraine and Japan, new countries appeared in this last analysis. 
The **USA**, which were second in the plots we have seen before, are now out of this list, symptom that the huge population may be a factor that influeced the first analysis.
* The **gender comparison** reflects the result previously reported from the **global trend over 100k population**. In every country of this list we had **more male suicides than female ones**:
  - If we look at the female rates, **Korea** seems to be the country with **most female suicides over 100k population**.
* Also, the **age group comparison** still reflects the result we've seen before. In each of the countries in this last list, **most of the rates occurred in the 75+ age group**. From this comparison we see also that:
  - **Korea** has by far the **highest rates within the "75+ years", "55-74 years", "35-54 years" and "25-34 years" age group**.
  - **Lithuania** and **Finland** are the only countries in this list that don't follow the global trends over 100k population. In fact, in those countries, the **"35-54 years" age group** is **the one with the highest rates**.
'''
)

#===================================================================

#4. Correlation
st.header('Correlation and Machine Learning')

st.write(
'''
In this paragraph we are gonna see the **correlation** between our data and we are gonna try to implement some basic **Machine Learning** algorithms.
\n **For this purpose we are gonna use the data until 2016 because, as we have stated before, from that year the values in the population columns start to get messy.**

'''
)

#Heatmap plot
if st.checkbox('Click to see the **heatmap**'):
    f, ax = plt.subplots(figsize=(12,10))
    
    sns.heatmap(suicide_df_until_2016.corr(), annot=True)
    
    st.write(f)
    st.caption('**Heatmap** that shows the correlation among our data (***for this chart we used the data until 2016, because, as we have stated before, from that year the values in the population column start to get messy***).')

#Writing the conclusions found by the heatmap analysis
st.write(
'''
Looking at the **heatmap** we see that the strongest correlation in our data is between the suicide rates and the **population**, with a value of **0.62**. Also, there's some correlation (*even though it's not so strong*) between the **GDP per capita** and the **year** column, with a value of 0.34.
'''
)

st.info('''
We can see that, although someone may think the opposite, **GDP per capita** **doesn't seem to affect the suicide rates** overall, as they have a correlation value that is equal to only **0,062**.
'''
)

# 1. GDP VS YEAR ANALYSIS + CLUSTERING
st.subheader('Analysis of the trend of the GDP per capita over the years and clustering example')

#GDP VS YEAR PLOTS
option3 = st.selectbox(
    'What plot you want to display?',
    ('Choose the plot','Line plot', 'Scatter plot'),
    key = 'selectbox3'
)
if option3 == 'Line plot':
    #lineplot gdp vs year
    f, ax = plt.subplots(figsize=(12,10))

    ax.set_title('Trend of the GDP per capita over the years', weight='bold')
    
    sns.lineplot(x= 'year', y='gdp_per_capita', data=suicide_df_until_2016)
    
    ax.set_ylabel('GDP per capita')
    ax.set_xlabel('Years')
    
    st.write(f)
elif option3 == 'Scatter plot':
    #scatterplot gdp vs year
    f, ax = plt.subplots(figsize=(12,10))
    
    ax.set_title('Trend of the GDP per capita over the years', weight='bold')
    
    plt.scatter(x='year', y='gdp_per_capita', data=suicide_df_until_2016)
    
    ax.set_ylabel('GDP per capita')
    ax.set_xlabel('Years')
    
    st.write(f)
else:
    st.write('')

st.write(
'''
From the following plots we can see how the **GDP per capita** had an overall **increasing trend** over the years.
\n Now, we are gonna try to implement a simple **clustering** example, using the **KMeans algorithm**, to determine three different GDP per capita groups: **low**, **medium** and **high**.
'''
)


#EXPANDER CLUSTERING MODEL

cluster_code_example = '''
#Import
from sklearn.cluster import KMeans

#Model
x = suicide_df_until_2016[['year', 'gdp_per_capita']]

km = KMeans(n_clusters=3, random_state=42)

y_pred = km.fit_predict(x)
'''

with st.expander('Clustering Model'):

    if st.checkbox('Click to see the code', key='clustercode'):
        st.code(cluster_code_example, language='python')

    #CLUSTERING ALGORITHM AND PLOTTING
    if st.button('Run the model'):
        with st.spinner('Training...'):

            #KMeans algorithm
            from sklearn.cluster import KMeans

            x = suicide_df_until_2016[['year', 'gdp_per_capita']]

            km = KMeans(n_clusters=3, random_state=42)

            y_pred = km.fit_predict(x)

            #Plotting
            f = plt.figure(figsize=(12,10))
            labels_cluster = ['Ordinary GDP per capita', 'Low GDP per capita', 'High GDP per capita']
            for i in range(3):
                plt.scatter(x.loc[y_pred==i, 'year'], x.loc[y_pred==i, 'gdp_per_capita'], label=labels_cluster[i]) 
            
            plt.xlabel('year')
            plt.ylabel('gdp_per_capita')
            
            plt.legend()
            
            st.write(f)
            st.caption('From the plot above we can see how the model finds the three clusters (***three levels of GDP per capita***) over the years (***for this model we used the data until 2016, without any estimated values***).')


#===========================================================================


# 2. SUICIDES VS POPULATION ANALYSIS
st.subheader('Analysis of the distribution of the rates among the population and Linear Regression example')

#scatterplot rates vs plot
if st.checkbox('Click to see the plot'):
    f, ax = plt.subplots(figsize=(12,10))
    
    ax.set_title('Distribution of the suicide rates among the population', weight='bold')
    
    plt.scatter(x='population', y='suicides_no', data=suicide_df_until_2016)
    
    ax.ticklabel_format(axis = 'x', style = 'plain')
    ax.set_ylabel('Total suicide rates')
    ax.set_xlabel('Population')
    
    st.write(f)

linear_reg_code_example ='''
#import
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#slider
test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)  

#model
feature = suicide_df_until_2016[['population']]
target = suicide_df_until_2016['suicides_no']

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=test_size, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
'''

#Linear Regression
with st.expander('Linear Regression Model'):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    if st.checkbox('Show the code', key='linreg_code'):
        st.code(linear_reg_code_example, language='python')

    #Slider
    test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)

    if st.button('Run the linear regression model'):
        with st.spinner('Training...'):
            feature = suicide_df_until_2016[['population']]
            target = suicide_df_until_2016['suicides_no']

            x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=test_size, random_state=42)

            model = LinearRegression()
            
            model.fit(x_train, y_train)
            
            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)

            #Plot
            fig = plt.figure(figsize=(12,10))
            plt.title('Linear Regression Model', weight='bold')

            #Scatter
            plt.scatter(x_test, y_test, color='blue')

            # Regression line
            plt.plot(x_test, y_pred, color='red')

            plt.xlabel('Population')
            plt.ylabel('Total suicide rates')
            plt.ticklabel_format(style = 'plain')

            st.write(fig)
            st.write('Mean Squared Error: ', mse)

#============================================================================================================================