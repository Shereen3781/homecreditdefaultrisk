## Home Credit Default Risk

Many people struggle to get loans, however due to insufficient information some untrustworthy lenders tend to be granted loans. 
This project is using data from kaggle about clients who get home loans. It aims to use of a variety of data to predict clients' repayment abilities. 
Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower clients to be successful.


## Most correlated features:

This is correlation matrix after deleting features with jigh percentage of missing values. There are still some features are highly correlated so i deleted one of each highly correlated pairs.

![image](https://github.com/Shereen3781/homecreditdefaultrisk/assets/110721883/20d63ca2-7816-4aae-9338-ff0027956713)

## Feature importances:

Random Forest Classifier turned out to the best model with roc_auc_score of 78%.

Two of these features are calculated by me to replace four features:
DAYS_EMPLOYED_PERCENT represnting the percentage of the days employed relative to the client's age, and
CREDIT_TERM represnting the length of the payment in months (since the annuity is the monthly amount due

![image](https://github.com/Shereen3781/homecreditdefaultrisk/assets/110721883/17d2a0a4-fa36-4b76-8088-cf09c7ccb72f)
