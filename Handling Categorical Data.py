
import pandas as pd
import numpy as np
import copy
%matplotlib inline

df_flights = pd.read_csv('flights.csv')
df_flights.head()

df_cat = df.select_dtypes(include = ['object']).columns ; df_cat
df_num = df.select_dtypes(exclude = ['object']).columns ; df_num 

print(df_flights.info())
df_flights.boxplot('dep_time','origin',rot = 30,figsize=(5,6))
cat_df_flights = df_flights.select_dtypes(include=['object']).copy()
cat_df_flights.head()
print(cat_df_flights.isnull().values.sum())
print(cat_df_flights.isnull().sum())
cat_df_flights = cat_df_flights.fillna(cat_df_flights['tailnum'].value_counts().index[0])
print(cat_df_flights.isnull().values.sum())
print(cat_df_flights['carrier'].value_counts())
print(cat_df_flights['carrier'].value_counts().count())

%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
carrier_count = cat_df_flights['carrier'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of Carriers')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()

labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()
counts = cat_df_flights['carrier'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()

replace_map = {'carrier': {'AA': 1, 'AS': 2, 'B6': 3, 'DL': 4, 'F9': 5, 'HA': 6, 'OO': 7 , 'UA': 8 , 'US': 9,'VX': 10,'WN': 11}}

labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()
replace_map_comp = {'carrier' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
print(replace_map_comp)
cat_df_flights_replace = cat_df_flights.copy()
cat_df_flights_replace.replace(replace_map_comp, inplace=True)
print(cat_df_flights_replace.head())
print(cat_df_flights_replace['carrier'].dtypes)

cat_df_flights_lc = cat_df_flights.copy()
cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].astype('category')
cat_df_flights_lc['origin'] = cat_df_flights_lc['origin'].astype('category')                                                              
print(cat_df_flights_lc.dtypes)

import time
%timeit cat_df_flights.groupby(['origin','carrier']).count() #DataFrame with object dtype columns
%timeit cat_df_flights_lc.groupby(['origin','carrier']).count() #DataFrame with category dtype columns

cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].cat.codes
cat_df_flights_lc.head() #alphabetically labeled from 0 to 10

cat_df_flights_specific = cat_df_flights.copy()
cat_df_flights_specific['US_code'] = np.where(cat_df_flights_specific['carrier'].str.contains('US'), 1, 0)
cat_df_flights_specific.head()

cat_df_flights_sklearn = cat_df_flights.copy()
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
cat_df_flights_sklearn['carrier_code'] = lb_make.fit_transform(cat_df_flights['carrier'])
cat_df_flights_sklearn.head() #Results in appending a new column to df

cat_df_flights_onehot = cat_df_flights.copy()
cat_df_flights_onehot = pd.get_dummies(cat_df_flights_onehot, columns=['carrier'], prefix = ['carrier'])
print(cat_df_flights_onehot.head())

cat_df_flights_onehot_sklearn = cat_df_flights.copy()
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb_results = lb.fit_transform(cat_df_flights_onehot_sklearn['carrier'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)
print(lb_results_df.head())

result_df = pd.concat([cat_df_flights_onehot_sklearn, lb_results_df], axis=1)
print(result_df.head())

cat_df_flights_ce = cat_df_flights.copy()
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['carrier'])
df_binary = encoder.fit_transform(cat_df_flights_ce)
df_binary.head()

encoder = ce.BackwardDifferenceEncoder(cols=['carrier'])
df_bd = encoder.fit_transform(cat_df_flights_ce)
df_bd.head()

dummy_df_age = pd.DataFrame({'age': ['0-20', '20-40', '40-60','60-80']})
dummy_df_age['start'], dummy_df_age['end'] = zip(*dummy_df_age['age'].map(lambda x: x.split('-')))
dummy_df_age.head()

dummy_df_age = pd.DataFrame({'age': ['0-20', '20-40', '40-60','60-80']})
def split_mean(x):
    split_list = x.split('-')
    mean = (float(split_list[0])+float(split_list[1]))/2
    return mean

dummy_df_age['age_mean'] = dummy_df_age['age'].apply(lambda x: split_mean(x))
dummy_df_age.head()

#Dealing with Categorical Features in Big Data with Spark
from pyspark import SparkContext
sc = SparkContext()
from pyspark.sql import SparkSession as spark
print(spark.catalog.listTables())
spark_flights = spark.read.format("csv").option('header',True).load('Downloads/datasets/nyc_flights/flights.csv',inferSchema=True)
spark_flights.show(3)
spark_flights.printSchema()
spark_flights.createOrReplaceTempView("flights_temp")
print(spark.catalog.listTables())
[Table(name=u'flights_temp', database=None, description=None, tableType=u'TEMPORARY', isTemporary=True)]
carrier_df = spark_flights.select("carrier")
carrier_df.show(5)
from pyspark.ml.feature import StringIndexer
carr_indexer = StringIndexer(inputCol="carrier",outputCol="carrier_index")
carr_indexed = carr_indexer.fit(carrier_df).transform(carrier_df)
carr_indexed.show(7)

carrier_df_onehot = spark_flights.select("carrier")
from pyspark.ml.feature import OneHotEncoder, StringIndexer
stringIndexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")
model = stringIndexer.fit(carrier_df_onehot)
indexed = model.transform(carrier_df_onehot)
encoder = OneHotEncoder(dropLast=False, inputCol="carrier_index", outputCol="carrier_vec")
encoded = encoder.transform(indexed)
encoded.show(7)


#Imputiong all columns by mode
for i in df.columns:
    df[i] = df[i].fillna(df[i].value_counts().index[0])

## This is a smoothong method which is amazing that someone should try.
for i in df.select_dtypes(include=['object']).columns:#Here you can provide the list of columns
    mn = df[i].mean()
    aggs = df.groupby(i)['LowDoc'].agg(['count','mean'])
    counts= aggs['count']
    means= aggs['mean']
    smooth = np.round(((counts * means + 100 * mn) / (counts + 100)),3)
    df[i] = df[i].map(smooth)


