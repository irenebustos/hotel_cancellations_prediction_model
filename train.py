# # Hotel reservations cancellations prediction

# Importing libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

from imblearn.over_sampling import SMOTE, RandomOverSampler
import pickle

# Load the dataset
df = pd.read_csv('hotel_reservations.csv')

#  Data Preparation
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df['total_people']  = df['no_of_adults'] + df['no_of_children']
df['price_per_adult'] = df['avg_price_per_room'] // df['no_of_adults']
df['price_per_person']  = df['avg_price_per_room'] // df['total_people']
df['has_prev_cancellations'] = df['no_of_previous_cancellations'] > 0
df['has_prev_bookings_not_cancelled'] = df['no_of_previous_bookings_not_canceled'] > 0
df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
df['have_children'] = df['no_of_children'] > 0
df['have_children'] = df['have_children'].astype('int')

def is_leap_year(year):
    return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
def adjust_for_feb_29(year, month, day):
    if month == 2 and day == 29 and not is_leap_year(year):
        return (month, 28)  
    return (month, day)

df['arrival_month'] = df['arrival_month'].apply(lambda x: f'{int(x):02d}')
df['arrival_date'] = df['arrival_date'].apply(lambda x: f'{int(x):02d}')
df[['arrival_month', 'arrival_date']] = df.apply(
    lambda row: adjust_for_feb_29(row['arrival_year'], row['arrival_month'], row['arrival_date']), axis=1,
    result_type='expand'
)
df['arrival_date_complete'] = pd.to_datetime(
    df[['arrival_year', 'arrival_month', 'arrival_date']].astype(str).agg('-'.join, axis=1),
    format='%Y-%m-%d', errors='coerce'
)

df['arrival_date_complete'] = df['arrival_date_complete'].fillna(pd.to_datetime('2018-02-28'))
df['arrival_date_complete'] = pd.to_datetime(df['arrival_date_complete'])

df['wday'] = df['arrival_date_complete'].dt.day_name()

# Feature selection and prep
df = df.drop(['booking_id','no_of_previous_cancellations','arrival_date','no_of_previous_cancellations', 'arrival_year',
              'no_of_previous_bookings_not_canceled','has_prev_cancellations', 'has_prev_bookings_not_cancelled',
               'price_per_adult','no_of_children','arrival_date_complete' ,'total_people'], axis=1)

df['booking_cancelled_flag'] = df['booking_status'].replace({'canceled': 1, 'not_canceled': 0})
df = df.drop(['booking_status'], axis=1)

df['arrival_month'] = df['arrival_month'].astype('int')
df['arrival_month'].value_counts()

# Train test split
categorical= ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'wday']
numerical = ['no_of_weekend_nights', 'no_of_week_nights', 'required_car_parking_space', 'lead_time', 'repeated_guest',  'price_per_person',
        'avg_price_per_room', 'no_of_special_requests','total_nights', 'arrival_month', 
              'no_of_adults', 'have_children']

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1, stratify=df['booking_cancelled_flag'])
print(len(df_full_train), len(df_test))

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1, stratify=df_full_train['booking_cancelled_flag'])
print(len(df_train), len(df_val), len(df_test))

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

y_train = df_train.booking_cancelled_flag.values
y_val = df_val.booking_cancelled_flag.values
y_test = df_test.booking_cancelled_flag.values
y_full_train = df_full_train.booking_cancelled_flag.values

del df_train['booking_cancelled_flag']
del df_val['booking_cancelled_flag']
del df_test['booking_cancelled_flag']
del df_full_train['booking_cancelled_flag']

# ## Model

def train_with_smote(df_train, y_train, params, categorical, numerical):
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)
    
    smote = SMOTE(random_state=42, sampling_strategy=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_resampled, label=y_resampled, feature_names=features)
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=200, 
        evals=[(dtrain, 'train')], 
        verbose_eval=5, 
        early_stopping_rounds=5
    )
    
    return dv, model

def predict(df, dv, model, categorical, numerical):
    data_dict = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(data_dict)
    features = list(dv.get_feature_names_out()) 
    dmatrix = xgb.DMatrix(X, feature_names=features)
    
    y_pred = model.predict(dmatrix)
    return y_pred

xgb_params = {
    'eta': 0.1,
    'max_depth': 12,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}

print('Training the final model with SMOTE...')
dv, model = train_with_smote(df_full_train, y_full_train, xgb_params, categorical, numerical)

y_pred_val = predict(df_val, dv, model, categorical, numerical)
y_pred_binary_val = (y_pred_val >= 0.5).astype(int)

print('Training the final model on the full dataset with SMOTE...')
dv, model = train_with_smote(df_full_train, y_full_train, xgb_params, categorical, numerical)

y_pred_test = predict(df_test, dv, model, categorical, numerical)
y_pred_binary_test = (y_pred_test >= 0.5).astype(int)

accuracy_test = accuracy_score(y_test, y_pred_binary_test)
precision_test = precision_score(y_test, y_pred_binary_test)
recall_test = recall_score(y_test, y_pred_binary_test)
f1_test = f1_score(y_test, y_pred_binary_test)
roc_auc_test = roc_auc_score(y_test, y_pred_test)

print(f'Test Accuracy: {accuracy_test * 100:.2f}%')
print(f'Test Precision: {precision_test * 100:.2f}%')
print(f'Test Recall: {recall_test * 100:.2f}%')
print(f'Test F1 Score: {f1_test * 100:.2f}%')
print(f'Test ROC AUC Score: {roc_auc_test * 100:.2f}%')

output_file = 'xgboost_model_booking_cancellation_smote.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model and DictVectorizer are saved to {output_file}')
