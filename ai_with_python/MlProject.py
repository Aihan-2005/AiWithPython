import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



#######################
# 1. load DATA

data = pd.read_csv("data/marketing_campaign.csv", sep="\t")


print(data.head())
print(data.info())
print(data.isnull().sum())

#######################
# 2. Copy DATA


df = data.copy()

#######################
# 3. Feature Engineering


df['Age'] = 2024 - df['Year_Birth']

df["Total_Spent"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
)

df["Total_Purchases"] = (
    df["NumDealsPurchases"] +
    df["NumWebPurchases"] +
    df["NumCatalogPurchases"] +
    df["NumStorePurchases"]
)

df["Children"] = df["Kidhome"] + df["Teenhome"]

df["Living_With"] = df["Marital_Status"].replace({
    "Single": "Alone",
    "Together": "Partner",
    "Married": "Partner",
    "Divorced": "Alone",
    "Widow": "Alone",
    "Alone": "Alone",
    "Absurd": "Alone",
    "YOLO": "Alone"
})

df["Family_Size"] = df["Living_With"].replace({
    "Alone": 1,
    "Partner": 2
}) + df["Children"]

df["Is_Parent"] = np.where(df["Children"] > 0, 1, 0)

df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate"
})

df["Campaign_Acceptance"] = (
    df["AcceptedCmp1"] +
    df["AcceptedCmp2"] +
    df["AcceptedCmp3"] +
    df["AcceptedCmp4"] +
    df["AcceptedCmp5"] +
    df["Response"]
)

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
last_date = df["Dt_Customer"].max()

df["Customer_For"] = (last_date - df["Dt_Customer"]).dt.days

df["Spending_Per_Purchase"] = df["Total_Spent"] / (df["Total_Purchases"] + 1)

df["Income_Per_Family_Member"] = df["Income"] / df["Family_Size"]



#######################
# 4. Drop Unnecssary Columsn


columns_to_drop = [
    'ID',
    'Year_Birth',
    'DT_Customer',
    'Marital_Status',
    'Z_CostContact',
    'Z_Revenue'
]


df = df.drop(columns=columns_to_drop,axis=1)


#######################
# 5. Handel Missing Values



df['Income'] = df['Income'].fillna((df['Income']).median())
df['Incom_per_Family_Member'] = df['Income_Per_Family_Member'].fillna(
    df['Income_Per_Family_Member'].median()
)


#######################
# 6. Handel Outliers


for col in ['Age','Income','Total_Spent']:
    lower =  df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower,upper)
    
    
    
#######################
# 7. Separate Numeric   and Categorical Columns


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols =  df.select_dtypes(exclude=['object']).columns.tolist()

print('Categorical columns:', categorical_cols)
print('Numeric columns:', numeric_cols)


#######################
#  8. Preprocessing

preprocessor = ColumnTransformer(
    transformers= [
        ('num',StandardScaler(),numeric_cols),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_threshold=False),categorical_cols)
    ]
)



X_processed = preprocessor.fit_transform(df)

feature_names = preprocessor.get_feature_names_out()

X_processed_df = pd.DataFrame(X_processed,columns=feature_names)

print(X_processed_df.head())


#######################
# .9 PCA for Visualization

pca = PCA.fit_transform(X_processed_df)

X_pca = pca.fit_transform(X_processed_df)

pca_df = pd.DataFrame(X_pca,columns=['PC1','PC2'])


print('Explained variance ratio:', pca.explained_variance_ratio_)
print('Total explained variance:', pca.explained_variance_ratio_.sum())


#######################
# 10. Elbow Method 


inertias = []
K_range = range(2,11)


for k in K_range:
    Kmeans =  KMeans(n_clusters=k,random_state=6,n_init=10)
    Kmeans.fit(X_processed_df)
    inertias.append(Kmeans.inertia_)
    
    
plt.figure(figsize=(8,5))
plt.plot(K_range,inertias,marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()



#######################
#   11. Silhouette Score


silhouette_score = []


for k in K_range:
    Kmeans = KMeans(n_clusters=k,random_state=6,n_init=10)
    labels = Kmeans.fit_predict(X_processed_df)
    score = silhouette_score(X_processed_df,labels)
    silhouette_score.append(score)
    
    
plt.figsize(figsize=(8,5))
plt.plot(K_range,silhouette_score,marker='o')
plt.xlabel('Number of Clusters')
plt.ylable('Silhoutte Score')
plt.title('Silhoutte Score for Different k')
plt.show()



best_k = list(K_range)[np.argmax(silhouette_score)]


print('Best K:',best_k)



#######################
# 12. Final KMeans Cluster


final_kmeans = KMeans(n_clusters=best_k,random_state=6,n_init=10)

clusters = final_kmeans.fit_predict(X_processed_df)

df['Cluster'] = clusters
pca_df['clusters'] = clusters



#######################
# 13. Visualize Cluster

plt.figure(figsize=(9,6))

sns.scatterplot(
    data= pca_df,
    x = 'PC1',
    y = 'PC2',
    hue = 'Cluster',
    palette='Set2'
)


plt.title('Customer segments Visualized with PCA')
plt.show()



#######################
# 14. Cluster Counts


print(df['Cluster'].value_counts())


plt.figure(figsize=(7,5))
sns.countplot(data=df,x='Cluster')
plt.title('Number of Customers in Each Cluster')
plt.show()




#######################
# 15. Cluster Profiling


cluster_profile = df.groupby('Cluster').mean(numeric_only=True)


important_cols = [
    'Age',
    'Income',
    'Total_Spent',
    'Total_Purchases',
    'Children',
    'Family_Size',
    'Campaign_Acceptance',
    'Customer_For',
    'Spending_Per_Purchase',
    'Income_Per_Family_Member'
]


print(cluster_profile[important_cols])


#######################
# 16. Distribution Plots

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Cluster", y="Income")
plt.title("Income Distribution by Cluster")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Cluster", y="Total_Spent")
plt.title("Total Spending by Cluster")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Cluster", y="Total_Purchases")
plt.title("Total Purchases by Cluster")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Cluster", y="Age")
plt.title("Age Distribution by Cluster")
plt.show()


#######################
# 17. Categorical Analysis

print(pd.crosstab(df['Cluster'],df['Education'],normalize='index'))

print(pd.crosstab(df['Cluster'],df['Living_With'],normalize='index'))


#######################
# 18. Save Final Data

df.to_csv('customer_segments_final.csv',index=False)

print('Final file saved successfully')
