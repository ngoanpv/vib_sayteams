import numpy as np 
import pandas as pd 

def generate_deposit_features(deposit_df, begin_month=1, end_month=12):
  deposit_df['MONTH'] = pd.DatetimeIndex(deposit_df['MONTH']).month
  deposit_feature_result = []
  customer_number_list = list(deposit_df[(deposit_df.MONTH >= begin_month) &
                          (deposit_df.MONTH <= end_month)].CUSTOMER_NUMBER.unique())
  for customer_number in tqdm(customer_number_list):
    deposit_features = {}
    deposit_features['CUSTOMER_NUMBER'] = customer_number
    customer_df = deposit_df[(deposit_df.CUSTOMER_NUMBER == customer_number)].sort_values("MONTH")
    if (customer_df['COUNT_CA_ACCT'] >= 1).any():
      deposit_features['HAS_CA_ACCOUNT'] = True
    else:
      deposit_features['HAS_CA_ACCOUNT'] = False

    if (customer_df['COUNT_TD_ACCT'] >= 1).any():
      deposit_features['HAS_TD_ACCOUNT'] = True
    else:
      deposit_features['HAS_TD_ACCOUNT'] = False
    
    deposit_features['MAX_MONTH_CA'] = customer_df[customer_df['AVG_CA_BALANCE'] 
                                                   == customer_df['AVG_CA_BALANCE'].max()].MONTH.tolist()[0]
    deposit_features['MIN_MONTH_CA'] = customer_df[customer_df['AVG_CA_BALANCE'] 
                                                   == customer_df['AVG_CA_BALANCE'].min()].MONTH.tolist()[0]
    deposit_features['MAX_MONTH_TD'] = customer_df[customer_df['AVG_TD_BALANCE'] 
                                                   == customer_df['AVG_TD_BALANCE'].max()].MONTH.tolist()[0]
    deposit_features['MIN_MONTH_TD'] = customer_df[customer_df['AVG_TD_BALANCE'] 
                                                   == customer_df['AVG_TD_BALANCE'].min()].MONTH.tolist()[0]
    
    diff_ca_list = customer_df['AVG_CA_BALANCE'].diff().tolist()
    diff_ca_list = diff_ca_list[1:]
    diff_ca_list = list(set(diff_ca_list))
    diff_td_list = customer_df['AVG_TD_BALANCE'].diff().tolist()
    diff_td_list = diff_td_list[1:]
    diff_td_list = list(set(diff_td_list))

    stable_ca = True
    stable_td = True
    if (len(diff_ca_list) != 1):
        stable_ca = False
    if (len(diff_td_list) != 1):
        stable_td = False
  
    deposit_features['STABLE_CA'] = stable_ca
    deposit_features['STABLE_TD'] = stable_td
    deposit_features['TD_MONTHS'] = customer_df[customer_df.COUNT_TD_ACCT > 0].shape[0]
    deposit_features['COUNT_CA_ACCT'] = customer_df['COUNT_CA_ACCT'].max()
    deposit_features['COUNT_TD_ACCT'] = customer_df['COUNT_TD_ACCT'].max()
    deposit_features['AVG_CA_BALANCE'] = customer_df['AVG_CA_BALANCE'].mean()
    deposit_features['AVG_TD_BALANCE'] = customer_df['AVG_TD_BALANCE'].mean()
    deposit_features['MAX_DIFF_CA_BALANCE'] = max(diff_ca_list) if len(diff_ca_list) > 0 else 0
    deposit_features['MIN_DIFF_CA_BALANCE'] = min(diff_ca_list) if len(diff_ca_list) > 0 else 0
    deposit_features['MAX_DIFF_TD_BALANCE'] = max(diff_td_list) if len(diff_td_list) > 0 else 0
    deposit_features['MIN_DIFF_TD_BALANCE'] = min(diff_td_list) if len(diff_td_list) > 0 else 0
    deposit_feature_result.append(deposit_features)
  df = pd.DataFrame(deposit_feature_result)
  return df  
