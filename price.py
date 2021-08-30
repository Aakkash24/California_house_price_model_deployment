import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def price_prediction(abc):
    housing_df = pd.read_csv("housing.csv")

    def outliers(df, fl):
        q1 = df[fl].quantile(0.25)
        q3 = df[fl].quantile(0.75)
        IQR = q3-q1
        lower_bound = q1 - 1.5*IQR
        upper_bound = q3 + 1.5*IQR
        ls = df.index[(df[fl] < lower_bound) | (df[fl] > upper_bound)]
        return ls

    index_list = []
    for x in ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']:
        index_list.extend(outliers(housing_df, x))

    def remove(df, ls):
        ls = sorted(set(index_list))
        df = df.drop(ls)
        return df

    housing_df = remove(housing_df, index_list)
    housing_df['rooms_per_household'] = housing_df['total_rooms'] / \
        housing_df['households']
    housing_df['bedrooms_per_households'] = housing_df['total_bedrooms'] / \
        housing_df['households']

    input_col = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
    target_col = 'median_house_value'

    inputs = housing_df[input_col].copy()
    targets = housing_df[target_col].copy()

    numeric_col = inputs.select_dtypes(np.number).columns.tolist()
    cate_col = inputs.select_dtypes('object').columns.tolist()

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer().fit(inputs[numeric_col])
    inputs[numeric_col] = imputer.transform(inputs[numeric_col])

    encoder = OneHotEncoder(
        sparse=False, handle_unknown='ignore').fit(inputs[cate_col])
    enc_col = list(encoder.get_feature_names(cate_col))
    enc_col = ['ocean_proximity_ OCEAN', 'ocean_proximity_INLAND',
               'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
    inputs[enc_col] = encoder.transform(inputs[cate_col])
    x = inputs[numeric_col+enc_col]

    from xgboost import XGBRegressor
    model = XGBRegressor(random_state=42, n_jobs=-1, max_depth=40,
                         n_estimators=200, learning_rate=0.1, booster='gbtree').fit(x, targets)
    model.fit(x, targets)

    col = numeric_col + enc_col
    a = abc
    inp = dict()
    for i in range(13):
        inp[col[i]] = a[i]
    inp = pd.DataFrame([inp])
    return model.predict(inp)
