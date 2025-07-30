
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_and_merge_data(training_path, output_path, hist_path):
    training_data = pd.read_csv(training_path)
    output = pd.read_csv(output_path)
    hist_data = pd.read_csv(hist_path)
    df = pd.merge(pd.merge(training_data, output, on='id'), hist_data, on='id', how='left')
    df.drop(columns=['id'], inplace=True, errors='ignore')
    return df

def clean_and_engineer(df):
    df['date_activ'] = pd.to_datetime(df['date_activ'], errors='coerce')
    df['tenure'] = (pd.to_datetime('2016-01-01') - df['date_activ']).dt.days / 365.25

    df['price_date'] = pd.to_datetime(df['price_date'], errors='coerce')
    agg_price_data = df.groupby(df.index).agg(
        mean_price_p1_var=('price_p1_var', 'mean'), std_price_p1_var=('price_p1_var', 'std'),
        mean_price_p2_var=('price_p2_var', 'mean'), std_price_p2_var=('price_p2_var', 'std'),
    ).reset_index(drop=True)
    df = df.drop_duplicates(subset=df.index).reset_index(drop=True)
    df = pd.concat([df, agg_price_data], axis=1)

    df['margin_x_tenure'] = df['net_margin'] * df['tenure']
    cols_to_drop = [col for col in df.columns if 'date' in col or 'price_p' in col or 'campaign_disc_ele' in col]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    for col in ['cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons']:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    if 'has_gas' in df.columns and df['has_gas'].dtype == 'object':
        df['has_gas'] = df['has_gas'].replace({'t': 1, 'f': 0})

    categorical_features = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    return df

def prepare_data(df, target_col='churn'):
    X = df.drop(columns=target_col, errors='ignore')
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    median_values = X_train.median()
    X_train.fillna(median_values, inplace=True)
    X_test.fillna(median_values, inplace=True)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test
