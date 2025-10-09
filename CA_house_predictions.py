#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_percentage_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


#preferences
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#making combined dataset
files = ["/Users/ethanc/CRMLSSold/CRMLSSold202408.csv", "/Users/ethanc/CRMLSSold/CRMLSSold202409.csv", "/Users/ethanc/CRMLSSold/CRMLSSold202410.csv", 
         "/Users/ethanc/CRMLSSold/CRMLSSold202411.csv", "/Users/ethanc/CRMLSSold/CRMLSSold202412.csv", "/Users/ethanc/CRMLSSold/CRMLSSold202501_filled.csv",
         "/Users/ethanc/CRMLSSold/CRMLSSold202502.csv", "/Users/ethanc/CRMLSSold/CRMLSSold202503.csv", "/Users/ethanc/CRMLSSold/CRMLSSold202504.csv",
         "/Users/ethanc/CRMLSSold/CRMLSSold202505.csv", "/Users/ethanc/CRMLSSold/CRMLSSold202506.csv", "/Users/ethanc/CRMLSSold/CRMLSSold202507.csv",
         "/Users/ethanc/CRMLSSold/CRMLSSold202508.csv"]
datasets = [pd.read_csv(f) for f in files]
df_all = pd.concat(datasets, ignore_index=True)

'''#exploring the data
#Basic data exploration
#print(df_all.info())
#print(dataset.head())
#print(df_all.columns)
#print(dataset.isnull().sum())
#print(dataset.nunique())

#Heatmap patterns
corr = dataset.corr(numeric_only=True)
dataset = dataset[dataset["ClosePrice"] < 6000000]

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt = ".2f")
plt.title("Feature Correlations with Close Price")
#plt.show()


#print(corr["ClosePrice"].sort_values(ascending=False))


#most relevant features plotted
sns.regplot(x="ListPrice", y="ClosePrice", data=dataset)
#plt.show()

sns.regplot(x="BuildingAreaTotal", y="ClosePrice", data=dataset)
#plt.show()

sns.regplot(x="LivingArea", y="ClosePrice", data=dataset)
#plt.show()

sns.regplot(x="BedroomsTotal", y="ClosePrice", data=dataset)
#plt.show()

sns.regplot(x="GarageSpaces", y="ClosePrice", data=dataset)
#plt.show()'''

#preprocessing (removing unnecessary columns)
remove_cols = [
    "BuyerAgentAOR", 
    "ListAgentAOR", 
    "Flooring", 
    "WaterfrontYN", 
    "BasementYN", 
    "OriginalListPrice", 
    "ListingKey", 
    "ListAgentEmail",  
    "ListAgentFirstName", 
    "ListAgentLastName", 
    "Latitude", 
    "Longitude", 
    "UnparsedAddress",  
    "ListPrice", 
    "DaysOnMarket", 
    "ListOfficeName", 
    "BuyerOfficeName", 
    "CoListOfficeName", 
    "ListAgentFullName", 
    "CoListAgentFirstName", 
    "CoListAgentLastName", 
    "BuyerAgentMlsId", 
    "BuyerAgentFirstName", 
    "BuyerAgentLastName", 
    "FireplacesTotal", 
    "AssociationFeeFrequency", 
    "AboveGradeFinishedArea", 
    "ListingKeyNumeric", 
    "MLSAreaMajor", 
    "TaxAnnualAmount", 
    "CountyOrParish", 
    "MlsStatus",
    "ElementarySchool",
    "AttachedGarageYN",
    "ParkingTotal",
    "BuilderName",
    "LotSizeAcres",
    "SubdivisionName",
    "BuyerOfficeAOR",
    "StreetNumberNumeric",
    "ListingId",
    "City",
    "TaxYear",
    "BuildingAreaTotal",
    "ContractStatusChangeDate",
    "ElementarySchoolDistrict",
    "CoBuyerAgentFirstName",
    "PurchaseContractDate",
    "ListingContractDate",
    "BelowGradeFinishedArea",
    "BusinessType",
    "StateOrProvince",
    "CoveredSpaces",
    "MiddleOrJuniorSchool",
    "FireplaceYN",
    "HighSchool",
    "Levels",
    "LotSizeDimensions",
    "LotSizeArea",
    "MainLevelBedrooms",
    "HighSchoolDistrict",
    "AssociationFee",
    "MiddleOrJuniorSchoolDistrict",
    "latfilled",
    "lonfilled"]

df = df_all.drop(columns=remove_cols)
df = df[(df["PropertyType"] == "Residential") & (df["PropertySubType"] == "SingleFamilyResidence")]
df = df.drop(columns=["PropertyType", "PropertySubType"])

#filling na values and converting to numerical
df["ViewYN"] = df["ViewYN"].fillna(0)
df["PoolPrivateYN"] = df["PoolPrivateYN"].fillna(0)

living_area_median = df["LivingArea"].median()
df["LivingArea"] = df["LivingArea"].fillna(living_area_median)

year_built_median = df["YearBuilt"].median()
df["YearBuilt"] = df["YearBuilt"].fillna(year_built_median)

bedrooms_median = df["BedroomsTotal"].median()
df["BedroomsTotal"] = df["BedroomsTotal"].fillna(bedrooms_median)

bathrooms_median = df["BathroomsTotalInteger"].median()
df["BathroomsTotalInteger"] = df["BathroomsTotalInteger"].fillna(bathrooms_median)

df["NewConstructionYN"] = df["NewConstructionYN"].fillna(0)

stories_median = df["Stories"].median()
df["Stories"] = df["Stories"].fillna(stories_median)

df["GarageSpaces"] = df["GarageSpaces"].fillna(0)

lot_size_median = df["LotSizeSquareFeet"].median()
df["LotSizeSquareFeet"] = df["LotSizeSquareFeet"].fillna(lot_size_median)

df["ViewYN"] = df["ViewYN"].astype(int)
df["PoolPrivateYN"] = df["PoolPrivateYN"].astype(int)
df["NewConstructionYN"] = df["NewConstructionYN"].astype(int)

df = df.dropna(subset=["PostalCode"])
df["PostalCode"] = df["PostalCode"].str[:5]
df["PostalCode"] = pd.to_numeric(df["PostalCode"])

df["CloseMonth"] = df["CloseDate"].str[5:7]
df["CloseMonth"] = pd.to_numeric(df["CloseMonth"])

df["CloseYear"] = df["CloseDate"].str[:4]
df["CloseYear"] = pd.to_numeric(df["CloseYear"])

df = df.dropna(subset=["ClosePrice"])
df = df.drop(columns="CloseDate")

df = df[(df["ClosePrice"]>=df["ClosePrice"].quantile(.005))&(df["ClosePrice"]<=df["ClosePrice"].quantile(.995))]

#split data for testing and training
df_test = df[(df["CloseMonth"] == 08.0) & (df["CloseYear"] == 2025)]

df_train = df[~(df["CloseMonth"] == 08.0)]

print(df.info())
'''
#tell what variables to use
X_train = df_train[["ViewYN", "PostalCode", "PoolPrivateYN", "LivingArea", "YearBuilt", "BathroomsTotalInteger", "BedroomsTotal", "Stories", "NewConstructionYN", "GarageSpaces", "LotSizeSquareFeet"]]
y_train = df_train["ClosePrice"]

X_test = df_test[["ViewYN", "PostalCode", "PoolPrivateYN", "LivingArea", "YearBuilt", "BathroomsTotalInteger", "BedroomsTotal", "Stories", "NewConstructionYN", "GarageSpaces", "LotSizeSquareFeet"]]
y_test = df_test["ClosePrice"]


#training diff models
#model = LinearRegression()

#training Random Forest Model
model = RandomForestRegressor(
    n_estimators = 500,
    max_depth=None,
    random_state=67,
    min_samples_split=2,
    min_samples_leaf=2,
    n_jobs=-1
)

model = DecisionTreeRegressor(
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=67
)

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=.1,
    max_depth=10,
    subsample=.8,
    random_state=67
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


#Gradient Boosting Regressor: R^2: 0.84549811013036, RMSE: 362255.6531145345, MAPE %: 14.180844539661678, Median MAPE: 8.842370969370764
#Random Forest Regressor results: R^2: 0.7981733729751737, RMSE: 414035.42832625436, MAPE %: 16.151178240782073, Median MAPE: 9.633472679394355
#DecisionTreeRegressor results: R^2: 0.6520970912127946, RMSE: 543597.3155486308, MAPE %: 26.505989890415556, Median MAPE: 17.966200485594783
#Linear Regressor results: R^2: 0.42680043995468564, RMSE: 697752.1729472455, MAPE %: 46.56291267159915, Median MAPE: 31.32234081762372
print("R^2:", r2_score(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE %:", mape * 100)
mape_values = np.abs((y_test - y_pred)/y_test) * 100
median_mape = np.median(mape_values)
print("Median MAPE:", median_mape)'''