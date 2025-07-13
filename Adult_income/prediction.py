import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from google.cloud import bigquery

# 指定專案 ID、資料集 ID 和表格 ID
project_id = 'incomepredict-67'
dataset_id = 'adult'
table_id = 'adult_data'
# 建立 BigQuery client 物件
client = bigquery.Client(project=project_id)
# 構建要查詢的完整表格名稱
table_ref = client.dataset(dataset_id).table(table_id)
# 構建 SQL 查詢語句以選取表格中的所有資料
query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"

# 執行查詢並將結果載入到 Pandas DataFrame
try:
    df = client.query(query).to_dataframe()
    print("成功從 BigQuery 讀取資料到 DataFrame！原始表單資料的前幾行是：")
    print(df.head()) # 顯示 DataFrame 的前幾行
except Exception as e:
    print(f"讀取 BigQuery 資料時發生錯誤：{e}")

df2 = df
df2['Marital_status'] = df2['Marital_status'].replace({
    'Married-civ-spouse': 'Married',
    'Married-spouse-absent': 'Married',
    'Married-AF-spouse': 'Married'
})

edu_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 
                   'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']
# 創建一個從學歷映射到數字的字典
education_mapping = {level: i + 1 for i, level in enumerate(edu_order)}
df2['education-num'] = None
# 新增參數：使用 .map() 方法將 'education' 欄位的值映射到 'education-num' 欄位
df2['education-num'] = df2['Education_Level'].map(education_mapping)

condition = (
    (df2['Relationship'].isin(['Husband', 'Wife'])) &
    (df2['Education_Level'].isin(['Bachelors', 'Assoc-voc', 'Assoc-acdm'])) &
    (df2['Occupation'].isin(['Exec-managerial'])) &
    (df2['Hours_per_week'] > 40) & (df2['Hours_per_week'] < 70)
)
df2['c2'] = np.where(condition, 1, 0)

# 新增權重 1
def is_married2(row):
    high_education = row['Education_Level'] in ['Masters', 'Prof-school', 'Doctorate']
    married_female = (row['Marital_status'] == 'Married') and (row['Sex'] == 'Female') and \
                     (row['Relationship'] == 'Wife') and high_education
    married_male = (row['Marital_status'] == 'Married') and (row['Sex'] == 'Male') and \
                   (row['Relationship'] == 'Husband') and high_education
    
    if married_female: return 2
    elif married_male: return 1
    elif row['c2'] == 1: return 3
    else: return 0
df2['married'] = None
df2['married'] = df2.apply(is_married2, axis=1)

# 將 occupation 欄位轉換為 Categorical 類型，並設定順序
occ_order = ['Exec-managerial', 'Prof-specialty', 'Tech-support', 'Sales', 'Craft-repair', 'Transport-moving', 'Protective-serv', 
             'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Handlers-cleaners', 'Other-service', 'Priv-house-serv']
df2['Occupation'] = pd.Categorical(df2['Occupation'], categories=occ_order, ordered=True)
df2['gender'] = df2['Sex'].map({'Male': 1, 'Female': 0})

df2['husband_wife'] = df2['Relationship'].map(lambda x: int((x =='Husband') | (x == 'Wife') ))
df2['edu_is_high'] = (df2['Education_Level'].isin(['Masters', 'Doctorate', 'Prof-school']))

# 判斷給定的一行資料（row）是否符合「從事專業特長或主管管理職位的男性」的條件，並返回一個整數值：1 代表符合條件，0 代表不符合。
def occupation_2(row):
    return int(row['Occupation'] in ['Prof-specialty','Exec-managerial'])
df2['occupation_2'] = df2.apply(occupation_2, axis=1)

# 判斷給定的一行資料（row）是否符合「中年男性」的條件，並返回一個整數值：1 代表是中年男性，0 代表不是。
def middle_age(row):
    good_age = range(35, 60)
    return int(good_age.count(row['Age']))
df2['middle_age'] = df2.apply(middle_age, axis=1)

def age_group(x):
    x = int(x)
    x = abs(x)
    if( 18 < x < 31 ):return 0
    if( 30 < x < 41 ):return 1
    if( 40 < x < 61 ):return 2
    if( 60 < x < 71 ):return 3
    else:return 4
df2['age_group'] = df2['Age'].apply(age_group)

def hour_group(x):
    x = int(x)
    x = abs(x)
    if( x < 35 ):return 0
    if( 35 <= x < 40 ):return 1
    if( 40 <= x < 45 ):return 2
    if( 45 <= x < 71 ):return 3
    else:return 4
df2['hour_group'] = df2['Hours_per_week'].apply(hour_group)

df2 = df2[['gender', 'edu_is_high', 'occupation_2', 'education-num', 'married', 'age_group',
          'middle_age', 'hour_group', 'husband_wife']]
loaded_model = joblib.load('random_forest_model.joblib')  # 載入模型
# 使用載入的模型進行預測
predictions = loaded_model.predict(df2)
predicted_labels = ['>50K' if pred == 0 else '≤50K' for pred in predictions]

# 將預測結果新增回 df
df = client.query(query).to_dataframe()
df['Predicted class'] = predicted_labels
print("\n包含預測結果的表單：")
print(df.head())
df.to_csv('Income prediction.csv', index=False, encoding='utf-8')
