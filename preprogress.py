import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
print('加载数据...')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# 数据探索
print('\n数据探索:')
print(f'训练集形状: {train_data.shape}')
print(f'测试集形状: {test_data.shape}')
print('\n训练集信息:')
print(train_data.info())
print('\n训练集描述性统计:')
print(train_data.describe())

# 检查缺失值
print('\n缺失值统计:')
print(train_data.isnull().sum())

# 数据预处理
print('\n数据预处理...')

# 合并训练集和测试集进行预处理
test_data['Listening_Time_minutes'] = np.nan  # 添加目标列以便合并
all_data = pd.concat([train_data, test_data], axis=0)

# 特征工程

# 1. 处理缺失值
# 对于数值型特征，使用中位数填充
all_data['Episode_Length_minutes'].fillna(all_data['Episode_Length_minutes'].median(), inplace=True)
all_data['Guest_Popularity_percentage'].fillna(all_data['Guest_Popularity_percentage'].median(), inplace=True)

# 2. 创建新特征
# 从播客名称提取长度作为特征
all_data['Podcast_Name_Length'] = all_data['Podcast_Name'].apply(lambda x: len(str(x)))

# 从剧集标题提取数字
def extract_episode_number(title):
    try:
        return int(''.join(filter(str.isdigit, str(title))))
    except:
        return 0

all_data['Episode_Number'] = all_data['Episode_Title'].apply(extract_episode_number)

# 创建时间段特征（早上、下午、晚上、夜间）
time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
all_data['Publication_Time_Encoded'] = all_data['Publication_Time'].map(time_mapping)

# 创建星期几特征
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
all_data['Publication_Day_Encoded'] = all_data['Publication_Day'].map(day_mapping)

# 情感编码
sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
all_data['Episode_Sentiment_Encoded'] = all_data['Episode_Sentiment'].map(sentiment_mapping)

# 分离回训练集和测试集
train_processed = all_data[all_data['Listening_Time_minutes'].notna()].copy()
test_processed = all_data[all_data['Listening_Time_minutes'].isna()].copy()

# 选择特征
numerical_features = [
    'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage',
    'Number_of_Ads', 'Episode_Number', 'Podcast_Name_Length',
    'Publication_Time_Encoded', 'Publication_Day_Encoded', 'Episode_Sentiment_Encoded'
]

categorical_features = ['Genre']

# 准备特征和目标变量
X = train_processed[numerical_features + categorical_features]
y = train_processed['Listening_Time_minutes']

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建预处理管道
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 模型训练和评估
print('\n模型训练和评估...')

# 随机森林模型
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict(X_val)
rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_pred))
rf_val_r2 = r2_score(y_val, rf_val_pred)

print(f'随机森林验证集RMSE: {rf_val_rmse:.4f}')
print(f'随机森林验证集R²: {rf_val_r2:.4f}')

# 梯度提升模型
gbr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

gbr_model.fit(X_train, y_train)
gbr_val_pred = gbr_model.predict(X_val)
gbr_val_rmse = np.sqrt(mean_squared_error(y_val, gbr_val_pred))
gbr_val_r2 = r2_score(y_val, gbr_val_pred)

print(f'梯度提升验证集RMSE: {gbr_val_rmse:.4f}')
print(f'梯度提升验证集R²: {gbr_val_r2:.4f}')

# 选择表现更好的模型
if gbr_val_rmse < rf_val_rmse:
    final_model = gbr_model
    print('\n选择梯度提升模型作为最终模型')
else:
    final_model = rf_model
    print('\n选择随机森林模型作为最终模型')

# 使用全部训练数据重新训练最终模型
final_model.fit(X, y)

# 特征重要性分析
if isinstance(final_model.named_steps['regressor'], RandomForestRegressor):
    feature_importances = final_model.named_steps['regressor'].feature_importances_
    
    # 获取特征名称（包括OneHotEncoder转换后的特征名称）
    preprocessor = final_model.named_steps['preprocessor']
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_features)
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(feature_importances)],
        'Importance': feature_importances
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    print('\n特征重要性:')
    print(importance_df.head(10))

# 在测试集上进行预测
print('\n在测试集上进行预测...')
X_test = test_processed[numerical_features + categorical_features]
test_predictions = final_model.predict(X_test)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_processed['id'],
    'Listening_Time_minutes': test_predictions
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print('\n提交文件已保存为 submission.csv')