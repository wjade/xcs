#XCS229II project

import os
os.chdir("/Users/yejiang/Desktop/Stanford ML class/project/code")

import numpy as np
import pandas as pd
from tabulate import tabulate
import sklearn
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#EDA
mydata = pd.read_csv (r'/Users/yejiang/Desktop/Stanford ML class/project/data/assembled-workers-compensation-claims-beginning-2000.csv')
col_names = mydata.columns.values.tolist()

mydata_backup = mydata

mydata = mydata[['Claim Identifier'
                 , 'Claim Type', 'District Name', 'Average Weekly Wage', 'Claim Injury Type', 'Age at Injury'
                 , 'Accident Date', 'Controverted Date', 'PPD Scheduled Loss Date', 'PPD Non-Scheduled Loss Date', 'First Appeal Date'
                 , 'WCIO Part Of Body Code', 'WCIO Nature of Injury Code', 'WCIO Cause of Injury Code', 'Alternative Dispute Resolution', 'Gender', 'Birth Year'
                 , 'Medical Fee Region', 'First Hearing Date', 'Closed Count', 'Attorney/Representative'
                 , 'Carrier Type', 'Accident', 'Occupational Disease', 'County of Injury']]


mydata.columns = ["Claim_Number", "Claim_Type", "District_Name", "Average_Weekly_Wage", "Injury_Type"
                                           , "Age_at_Injury", "Accident_Date", "Controverted_Date", "PPD_Scheduled_Loss_Date"
                                           , "PPD_Non-Scheduled_Loss_Date", "First_Appeal_Date"
                                           , "WCIO_Part", "WCIO_Nature", "WCIO_Cause"
                                           , "Alternative_Dispute_Res", "Gender", "Birth_Year"
                                           , "Med_Fee_Region", "First_Hearing_Date", "Closed_Count"
                                           , "Target", "Carrier_Type", "Workplace_Accident", "Occupational_Disease", "County_Injury"]

mydata.dtypes

#print summary for numerical variables
mydata.describe()

mydata['Accident_Date'] = pd.to_datetime(mydata['Accident_Date'], format = '%Y-%m-%d')
mydata['Controverted_Date'] = pd.to_datetime(mydata['Controverted_Date'], format = '%Y-%m-%d')
mydata['PPD_Scheduled_Loss_Date'] = pd.to_datetime(mydata['PPD_Scheduled_Loss_Date'], format = '%Y-%m-%d')
mydata['PPD_Non-Scheduled_Loss_Date'] = pd.to_datetime(mydata['PPD_Non-Scheduled_Loss_Date'], format = '%Y-%m-%d')
mydata['First_Appeal_Date'] = pd.to_datetime(mydata['First_Appeal_Date'], format = '%Y-%m-%d')
mydata['First_Hearing_Date'] = pd.to_datetime(mydata['First_Hearing_Date'], format = '%Y-%m-%d')


mydata['WCIO_Part'] = mydata['WCIO_Part'].fillna(0)
mydata['WCIO_Nature'] = mydata['WCIO_Nature'].fillna(0)
mydata['WCIO_Cause'] = mydata['WCIO_Cause'].fillna(0)

mydata['WCIO_Part'] = pd.Categorical(mydata['WCIO_Part'])
mydata['WCIO_Nature'] = pd.Categorical(mydata['WCIO_Nature'])
mydata['WCIO_Cause'] = pd.Categorical(mydata['WCIO_Cause'])


#check distribution of target variable
mydata['Target'].value_counts()

#sample claims w/o attorney rep from accident date < Dec 01, 2018(used >= Dec 01, 2016 to Dec 01, 2018 to minimize deterioration in data consistency because of time)
sample1 = mydata.loc[(mydata["Accident_Date"] >= '2016-12-01') & (mydata["Accident_Date"] < '2018-12-01' )]
sample1['Target'].value_counts()
sample1 = sample1.loc[sample1['Target'] == 'N']#453082 rows

#sample claims w attorney rep from accident date between Dec 01, 2018 and Dec 01, 2020
sample2 = mydata.loc[(mydata["Accident_Date"] >= '2018-12-01') & (mydata["Accident_Date"] < '2020-12-01' )]
sample2['Target'].value_counts()
sample2 = sample2.loc[sample2['Target'] == 'Y'] #96038 rows

#sample 97000 rows from sample 1(make sure 0/1 in target variable is balanced)
sample1_subset = sample1.sample(97000)

sample1_subset = sample1_subset.append(sample2)

data_cleaned = sample1_subset

#create "difference between dates" variable
data_cleaned['Days_btw_Controverted_Accident'] = data_cleaned['Controverted_Date'] - data_cleaned['Accident_Date']
data_cleaned['Days_btw_PPD_Scheduled_Accident'] = data_cleaned['PPD_Scheduled_Loss_Date'] - data_cleaned['Accident_Date'] 
data_cleaned['Days_btw_PPD_Non_Scheduled_Accident'] = data_cleaned['PPD_Non-Scheduled_Loss_Date'] - data_cleaned['Accident_Date'] 
data_cleaned['Days_btw_1stAppeal_Accident'] = data_cleaned['First_Appeal_Date'] - data_cleaned['Accident_Date'] 
data_cleaned['Days_btw_1stHearing_Accident'] = data_cleaned['First_Hearing_Date'] - data_cleaned['Accident_Date']

#drop un-needed variables
data_cleaned = data_cleaned.drop(['Accident_Date', 'Controverted_Date', 'PPD_Scheduled_Loss_Date', 'PPD_Non-Scheduled_Loss_Date', 'First_Appeal_Date', 'First_Hearing_Date'], axis = 1)

#clean the difference between dates variable
data_cleaned['Days_btw_Controverted_Accident'] = data_cleaned['Days_btw_Controverted_Accident']/ np.timedelta64(1, 'D')
data_cleaned['Days_btw_PPD_Scheduled_Accident'] = data_cleaned['Days_btw_PPD_Scheduled_Accident']/ np.timedelta64(1, 'D')
data_cleaned['Days_btw_PPD_Non_Scheduled_Accident'] = data_cleaned['Days_btw_PPD_Non_Scheduled_Accident']/ np.timedelta64(1, 'D')
data_cleaned['Days_btw_1stAppeal_Accident'] = data_cleaned['Days_btw_1stAppeal_Accident']/ np.timedelta64(1, 'D')
data_cleaned['Days_btw_1stHearing_Accident'] = data_cleaned['Days_btw_1stHearing_Accident']/ np.timedelta64(1, 'D')


data_cleaned['Days_btw_Controverted_Accident'] = data_cleaned['Days_btw_Controverted_Accident'].astype(float)
data_cleaned['Days_btw_PPD_Scheduled_Accident'] = data_cleaned['Days_btw_PPD_Scheduled_Accident'].astype(float)
data_cleaned['Days_btw_PPD_Non_Scheduled_Accident'] = data_cleaned['Days_btw_PPD_Non_Scheduled_Accident'].astype(float)
data_cleaned['Days_btw_1stAppeal_Accident'] = data_cleaned['Days_btw_1stAppeal_Accident'].astype(float)
data_cleaned['Days_btw_1stHearing_Accident'] = data_cleaned['Days_btw_1stHearing_Accident'].astype(float)

#convert categorical variables to dummy variables
data_cleaned = pd.get_dummies(data_cleaned, columns=['Claim_Type', 'District_Name', 'Injury_Type', 'Alternative_Dispute_Res','Gender', 'Med_Fee_Region', 'Carrier_Type', 'Workplace_Accident', 'Occupational_Disease', 'County_Injury'])

#train-dev-test split
Train, Dev = train_test_split(data_cleaned, test_size=0.2, random_state=1)
Train, Test = train_test_split(Train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

#clustering on train and apply it to dev and test
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
for_cluster = Train[['WCIO_Part', 'WCIO_Nature', "WCIO_Cause"]]

#do not run the code commented out below, will take long
#distortions = []
#K = range(50, 1000, 50)
#for k in K:
 #   kmeanModel = KMeans(n_clusters=k).fit(for_cluster)
 #   kmeanModel.fit(for_cluster)
 #   distortions.append(sum(np.min(cdist(for_cluster, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / for_cluster.shape[0])

# Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()

#turned out clusters is the best, therefore the model below --
k = 600
kmeanModel = KMeans(n_clusters = k).fit(for_cluster)

#predict the cluster on dev and test
Train_kmeans = kmeanModel.predict(Train[['WCIO_Part', 'WCIO_Nature', "WCIO_Cause"]])
Dev_kmeans = kmeanModel.predict(Dev[['WCIO_Part', 'WCIO_Nature', "WCIO_Cause"]])
Test_kmeans = kmeanModel.predict(Test[['WCIO_Part', 'WCIO_Nature', "WCIO_Cause"]])

Train['Cluster'] = Train_kmeans
Dev['Cluster'] = Dev_kmeans
Test['Cluster'] = Test_kmeans

Train = Train.drop(['WCIO_Part', 'WCIO_Nature', "WCIO_Cause"], axis = 1)
Dev = Dev.drop(['WCIO_Part', 'WCIO_Nature', "WCIO_Cause"], axis = 1)
Test = Test.drop(['WCIO_Part', 'WCIO_Nature', "WCIO_Cause"], axis = 1)


# split data into X and y
X_train = Train.drop(['Claim_Number', 'Target'], axis=1)
Y_train = Train['Target']

X_dev = Dev.drop(['Claim_Number', 'Target'], axis=1)
Y_dev = Dev['Target']

X_test = Test.drop(['Claim_Number', 'Target'], axis=1)
Y_test = Test['Target']
