## PROJECT : SALARY PREDİCTİON WİTH MACHİNE LEARNİNG


# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan
# beyzbol oyuncularının maaş tahminleri için bir makine öğrenmesi modeli geliştirilebilir mi?



### GEREKLİ KÜTÜPHANE VE FONKSİYONLAR  ###

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate,validation_curve

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 750)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)



### GELİŞMİŞ FONKSİYONEL KEŞİFCİ VERİ ANALİZİ (ADVANCED FUNCTİONAL EDA)  ###


#### 1. Genel Resim ####

df = pd.read_csv(f"C:/Users/duran/PycharmProjects/Hitters_Salary_Prediction/datasets/hitters.csv")

def check_df(dataframe, head=5):
    print("################ SHAPE #################")
    print(dataframe.shape)

    print("################ TYPES #################")
    print(dataframe.dtypes)

    print("################ HEAD #################")
    print(dataframe.head(head))

    print("################ TAİL #################")
    print(dataframe.tail(head))

    print("################ NA #################")
    print(dataframe.isnull().sum())

    print("################ QUANTİLES #################")
    print(dataframe.describe([0,0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



from scipy import stats

def show_salary_skewness(df):
    # Salary sütunundaki eksik değerleri kaldırma
    salary = df['Salary'].dropna()

    # Çarpıklık değerini hesaplama
    skewness = salary.skew()

    # Histogram ve yoğunluk grafiği çizme
    plt.figure(figsize=(10, 6))
    sns.histplot(salary, kde=True)
    plt.title(f'Salary Distribution (Skewness: {skewness:.2f})')
    plt.xlabel('Salary (thousands)')
    plt.ylabel('Frequency')

    # Ortalama ve medyanı gösterme
    plt.axvline(salary.mean(), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {salary.mean():.2f}')
    plt.axvline(salary.median(), color='g', linestyle='dashed', linewidth=2, label=f'Median: {salary.median():.2f}')
    plt.legend()

    # İstatistiksel bilgileri yazdırma
    print(f"Skewness: {skewness:.2f}")
    print(f"Mean: {salary.mean():.2f}")
    print(f"Median: {salary.median():.2f}")
    print(f"Standard Deviation: {salary.std():.2f}")
    print(f"Minimum: {salary.min():.2f}")
    print(f"Maximum: {salary.max():.2f}")

    # Shapiro-Wilk normallik testi
    _, p_value = stats.shapiro(salary)
    print(f"Shapiro-Wilk Test p-value: {p_value:.4f}")

    plt.show()


show_salary_skewness(df)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, sayısal ve kategorik gibi davranan ama sayısal olmayan değişkenleri ayırır.

    """

    # Kategorik değişkenleri seçme
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Sayısal değişkenleri seçme
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # Sonuçları yazdırma
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


# Kullanım örneği:
cat_cols, num_cols, cat_but_car = grab_col_names(df)



#### 2. Kategorik Değişken Analizi (Analysis of Categorical veriables) ####

def cat_sumary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100* dataframe[col_name].value_counts()/ len(dataframe)}))

    print('#############################################')

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block= True)

for col in cat_cols:
    cat_sumary(df, col, plot = True)


#### 3. Sayısal Veri Analizi (Analysis of Numerical Veriables) ####

def num_sumary(dataframe, numerical_col, plot = False):

    quantiless = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,0.99]

    print(dataframe[numerical_col].describe(quantiless).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block= True)


for col in num_cols:
    num_sumary(df, col, plot = True)

#### 4. Hedef Değişken Analizi (Analysis of Target Veriables) ####

def target_summary(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary(df, "Salary", col)


#### 5. Korelasyon Analizi (Analysis of Corelation) ####

print(df[num_cols].corr(method="spearman"))

fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(df[num_cols].corr(),annot= True,linewidths=5, ax=ax)
plt.show()


def find_corelation(dataframe, numeric_cols, corr_limits=0.50):
    high_corelation = []
    low_corelation = []
    for col in numeric_cols:
        if col == "Salary":
            pass
        else:
            corelation = dataframe[[col, "Salary"]].corr().loc[col, "Salary"]
            print(col, corelation)

            if abs(corelation) > corr_limits:
                high_corelation.append(col + " : " + str(corelation))
            else:
                low_corelation.append(col + " : " + str(corelation))
    return low_corelation, high_corelation


low_corrs, high_corrs = find_corelation(df, num_cols)
print("#################################################################")
print(low_corrs)
print("#################################################################")
print(high_corrs)


#### 6. Outliers (Aykırı Değerler) ####

sns.boxplot(x=df["Salary"], data= df)
plt.show()


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


def check_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_treshold(dataframe, veriable):
    low_limit, up_limit = outlier_thresholds(dataframe, veriable)
    dataframe.loc[dataframe[veriable] < low_limit, veriable] = low_limit
    dataframe.loc[dataframe[veriable] > up_limit, veriable] = up_limit


for col in num_cols:
    print(col, check_outliers(df, col))
print("#############################################")

for col in num_cols:
    if check_outliers(df, col):
        replace_with_treshold(df, col)



#### 7. Missing Values (Eksik Değerler) ####

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat(objs=[n_miss, np.round(ratio, decimals=2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

# Örnek kullanım
missing_values_table(df)


# Eksik veri analizine uygun olarak 3 farklı yöntem kullanılabilir.
df1 = df.copy()
df1.head()
cat_cols , num_cols, cat_but_car = grab_col_names(df1)

# method = int(input("Eksik veri için hangi yöntemi uygulamak istersiniz? (1/2/3): "))

from sklearn.impute import KNNImputer


def eksik_veri_doldur(dataframe, method):
    df1 = dataframe.copy()

    cat_cols, num_cols, cat_but_car = grab_col_names(df1)

    if method == 1:
        dff = pd.get_dummies(df1[cat_cols + num_cols], drop_first=True)
        scaler = RobustScaler()
        dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
        imputer = KNNImputer(n_neighbors=5)
        dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
        dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
        df1 = dff

        pass

    elif method == 2:
        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "A") & (df1["Division"] == "E"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].mean()["A", "E"]

        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "A") & (df1["Division"] == "W"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].mean()["A", "W"]

        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "N") & (df1["Division"] == "E"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].mean()["N", "E"]

        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "N") & (df1["Division"] == "W"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].mean()["N", "W"]
        pass
    elif method == 3:
        # Drop NA
        # Eksik değer içeren tüm satırları silme
        df1.dropna(inplace=True)
        pass
    return df1


df1 = eksik_veri_doldur(df, method=1)
print("********************************************************")
print(df1.isnull().sum())


#### 8. Feature Exraction (Özellik çıkarımı) ####

new_num_cols = [col for col in num_cols if col!= "Salary"]
df1[new_num_cols] = df1[new_num_cols]+ 0.0000000001

df1["Hits_Success"] = (df1["Hits"] / df1["AtBat"]) * 100
df1["NEW_RBI"] = df1["RBI"] / df1["CRBI"]
df1["NEW_Walks"] = df1["Walks"] / df1["CWalks"]
df1["NEW_PutOuts"] = df1["PutOuts"] / df1["Years"]
df1["NEW_Hits"] = df1["Hits"] / df1["CHits"] + df1["Hits"]
df1["NEW_CRBI*CATBAT"] = df1["CRBI"] * df1["CAtBat"]
df1["NEW_CHits"] = df1["CHits"] / df1["Years"]
df1["NEW_CHmRun"] = df1["CHmRun"] / df1["Years"]
df1["NEW_CRuns"] = df1["CRuns"] / df1["Years"]
df1["NEW_CHits"] = df1["CHits"] * df1["Years"]
df1["NEW_RW"] = df1["RBI"] * df1["Walks"]
df1["NEW_CH_CB"] = df1["CHits"] / df1["CAtBat"]
df1["NEW_CHm_CAT"] = df1["CHmRun"] / df1["CAtBat"]
df1["NEW_Diff_Atbat"] = df1["AtBat"] - (df1["CAtBat"] / df1["Years"])
df1["NEW_Diff_Hits"] = df1["Hits"] - (df1["CHits"] / df1["Years"])
df1["NEW_Diff_HmRun"] = df1["HmRun"] - (df1["CHmRun"] / df1["Years"])
df1["NEW_Diff_Runs"] = df1["Runs"] - (df1["CRuns"] / df1["Years"])
df1["NEW_Diff_RBI"] = df1["RBI"] - (df1["CRBI"] / df1["Years"])
df1["NEW_Diff_Walks"] = df1["Walks"] - (df1["CWalks"] / df1["Years"])

print(df1.columns)



df1["Salary"].isnull().sum()


#### 9. One-Hot Encoding ####

cat_cols, num_cols, cat_but_car = grab_col_names(df1)


print(num_cols)
print(cat_cols)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols ,drop_first=drop_first)
    for col in dataframe.columns:
        if dataframe[col].dtypes == 'bool':
            dataframe[col] = dataframe[col].astype(int)
    return dataframe

df1 = one_hot_encoder(df1, cat_cols, drop_first=True)

df1.head()


###################### MODELLİNG #########################

df1.dropna(inplace=True)
df1.isnull().sum()

y = df1["Salary"]
X = df1.drop("Salary", axis=1)

X.shape
y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=46)


from sklearn.linear_model import LinearRegression

# Model Evalutaion for Linear Regression
linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
y_pred = model.predict(X_train)
lin_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("Linear Regression Train RMSE:" , "{:,.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred))))



lin_train_r2 = linreg.score(X_train, y_train)
print("Linear Regression Train R2:", "{:,.3f}".format(linreg.score(X_train, y_train)))


model = linreg.fit(X_train, y_train)
y_pred = model.predict(X_test)
lin_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regression Test RMSE:" , "{:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))



lin_test_r2 = linreg.score(X_test, y_test)
print("Linear Regression Test R2:", "{:,.3f}".format(linreg.score(X_test, y_test)))



# Test part regplot
g = sns.regplot(x= y_test, y= y_pred, scatter_kws={'color': 'b','s': 5},
                ci=False, color='r')
g.set_title(f'Test Model R2 :  =  {linreg.score(X_test, y_test):.3f}')
g.set_ylabel('Predicted Salary')
g.set_xlabel('Salary')
plt.xlim(-5,2700)
plt.ylim(bottom = 0)
plt.show(block=True)


# Cross Validation score
print("Linear Regression Cross_Val_Score: ", "{:,.3f}".format(np.mean(np.sqrt(-cross_val_score(model,
                                                                                               X,
                                                                                               y,
                                                                                               cv=10,
                                                                                               scoring='neg_mean_squared_error')))))

# Bağımsız dfeğişkenin bağımlı değişkene etkisi

# EKK - en küçük kareler yöntemi
# OLS for Linear Regression
import statsmodels.api as sm
from statsmodels.formula.api import ols

# adding a constant to the model
X_train_sm = sm.add_constant(X_train)

# fitting the model using statsmodels
model_sm = sm.OLS(y_train,X_train_sm).fit()

# getting the summary of the regression model
model_summary = model_sm.summary()
print(model_summary)














