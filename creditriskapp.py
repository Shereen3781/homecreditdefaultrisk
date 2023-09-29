import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(layout= 'wide')

app_train = pd.read_csv('application_train1.csv')

row0_spacer1, row0_1, row0_spacer2, row0_2 = st.columns((.1, 1.3, .1, 1.3))
with row0_1:
    st.title('Home Credit Default Risk')
    st.subheader('App by: Shereen Saleh')
with row0_2:
    image = Image.open('about-us-home-credit.jpg')
    st.image(image, width=300)

st.markdown("This project is using data from kaggle about clients who get home loans.\
    It aims to use of a variety of data to predict clients' repayment abilities.\
    Doing so will ensure that clients capable of repayment are not rejected and that loans\
    are given with a principal, maturity, and repayment calendar that will empower clients\
    to be successful.")

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 30px;">Distribution of multiple features with TARGET</p>'
st.markdown(new_title, unsafe_allow_html=True)
row4_spacer1, row4_1, row4_spacer2, row4_2 = st.columns((.2, 7.1, .1, 7))
with row4_1:
    option5 = st.selectbox('selsect an option' , ['AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_EMPLOYED','DAYS_BIRTH', 'EXT_SOURCE_2', 'EXT_SOURCE_3','CNT_CHILDREN'],key = "tap_sec5")
    t1 = app_train.loc[app_train['TARGET'] != 0]
    t0 = app_train.loc[app_train['TARGET'] == 0]
    fig = plt.figure(figsize=(7,4))
    sns.kdeplot(t1[option5],bw_method=0.05, label="TARGET = 1")
    sns.kdeplot(t0[option5],bw_method=0.05,label="TARGET = 0")
    plt.ylabel('Density plot', fontsize=12)
    plt.legend()
    st.pyplot(fig)
with row4_2:
    st.markdown('Some numerical features that seems to have relationship to the likelihood of an applicant to repay a loan are:')
    st.markdown('Loan annuity,')
    st.markdown('The price of the goods for which the loan is given,')
    st.markdown('Period for which the client is employed,')
    st.markdown('Normalized score from external data source, and number of children the client has, and')
    st.markdown('Number of children the client has')

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 30px;">Characteristics of people who not repay their loan</p>'
st.markdown(new_title, unsafe_allow_html=True)
option2 = st.selectbox('selsect an option' , ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR', 'REG_REGION_NOT_LIVE_REGION','NAME_EDUCATION_TYPE'],key = "tap_sec2")
cat_perc=app_train[[option2, 'TARGET']].groupby([option2],as_index=False).mean().sort_values(by='TARGET', ascending=False)
global_mean = app_train.TARGET.mean()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
s=sns.countplot(x=option2, data=app_train, ax=ax1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
s =sns.barplot(x = option2, y = "TARGET" , data = cat_perc , ax = ax2)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
s.axhline(global_mean, linewidth=3, color='b')
plt.text(0, global_mean - 0.01, "global_mean", color='black', weight='semibold')
st.pyplot(fig)

st.markdown('Revolving loans are just a small fraction from the total number of loans. But compared with their frequency, a larger amount of Revolving loans, are not repaid.')
st.markdown('The number of female clients is almost double the number of male clients. But, males have a higher chance of not repaying their loans (almost 10%), comparing with women (almost 7%).')
st.markdown('Client owns a car are almost a half of the ones that doesn't own one. But they have a lower chance of not repaying their loans than the ones that doesn't own a car.')
st.markdown('Very few people are registered in not live. Generally, not repayment rate is slightly larger for these cases than in the rest.')
st.markdown('Majority of the clients have Secondary / secondary special education, followed by clients with Higher education. Only a very small number having an academic degree.The Lower secondary category, although rare, have the largest rate of not returning the loan (11%).\
The people with Academic degree have less than 2% not-repayment rate.')

option3 = st.selectbox('selsect an option' , ['ORGANIZATION_TYPE','NAME_INCOME_TYPE'],key = "tap_sec3")
cat_perc=app_train[[option3, 'TARGET']].groupby([option3],as_index=False).mean().sort_values(by='TARGET', ascending=False)
global_mean = app_train.TARGET.mean()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
s=sns.countplot(x=option3, data=app_train, ax=ax1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
s =sns.barplot(x = option3, y = "TARGET" , data = cat_perc , ax = ax2)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
s.axhline(global_mean, linewidth=3, color='b')
plt.text(0, global_mean - 0.01, "global_mean", color='black', weight='semibold')
fig.subplots_adjust(hspace =0.6)
st.pyplot(fig)

st.markdown('Oraganizations with highest percent of loans not repaid are Transport: type 3 (16%), Industry: type 13 (13.5%), Industry: type 8 (12.5%) and Restaurant (less than 12%).')
st.markdown('Most of clients have income from Working, followed by Commercial associate, Pensioner and State servant.\
Clients with maternity leave have not-repayment rates of almost 40%, followed by Unemployed (37%). The rest of types of incomes have not-repayment rates of 10%.')


new_title = '<p style="font-family:sans-serif; color:Green; font-size: 30px;">Missing values statistics</p>'
st.markdown(new_title, unsafe_allow_html=True)
total = app_train.isnull().sum().sort_values(ascending = False)
percent = ((app_train.isna().sum()/len(app_train))*100).sort_values(ascending = False)
missing_values=pd.concat([total, percent], axis=1, keys=['Total_na', 'Percent_na']).reset_index()
ms= missing_values[missing_values["Percent_na"] > 0]
fig,ax =plt.subplots(figsize=(15,10))
plt.xticks(rotation=90)
sns.barplot(x=ms.index, y=ms["Percent_na"],color="green",alpha=0.8)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
st.pyplot(fig)

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 30px;">Most correlated features</p>'
st.markdown(new_title, unsafe_allow_html=True)
app_train1 = pd.read_csv('app_train.csv')
numerical_col = app_train1.select_dtypes(exclude = 'O').columns
corrmat = app_train1[numerical_col].corr()
top_corr_features = corrmat.index[abs(corrmat["TARGET"])>=0.03]
fig=plt.figure(figsize=(20,10))
sns.heatmap(app_train1[top_corr_features].corr(),annot=True,cmap='Blues', fmt='.2f')
st.pyplot(fig)

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 30px;">Feature Importance</p>'
st.markdown(new_title, unsafe_allow_html=True)
features_imp = pd.read_csv('features_imp.csv')
features_imp1=features_imp.head(10)
fig=plt.figure(figsize=(12,8))
sns.barplot(x=features_imp1['Importance'], y=features_imp1['Features'], color="green")
st.pyplot(fig)
