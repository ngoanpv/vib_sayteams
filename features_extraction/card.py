import numpy as np 
import pandas as pd 

def generate_card_features(card_df, begin_month=1, end_month=12):
  card_df['MONTH'] = pd.DatetimeIndex(card_df['MONTH']).month
  card_feature_result = []
  customer_number_list = list(card_df[(card_df.MONTH >= begin_month) &
                          (card_df.MONTH <= end_month)].CUSTOMER_NUMBER.unique())
  for customer_number in tqdm(customer_number_list):
    card_features = {}
    card_features['CUSTOMER_NUMBER'] = customer_number
    customer_df = card_df[(card_df.CUSTOMER_NUMBER == customer_number)].sort_values("MONTH")
    if (customer_df['COUNT_CREDITCARD'] >= 1).any():
      card_features['HAS_CREDITCARD'] = True
    else:
      card_features['HAS_CREDITCARD'] = False

    if (customer_df['COUNT_DEBITCARD'] >= 1).any():
      card_features['HAS_DEBITCARD'] = True
    else:
      card_features['HAS_DEBITCARD'] = False
    
    card_features['MAX_CREDITCARD_NO'] = customer_df['COUNT_CREDITCARD'].max()
    card_features['MAX_DEBITCARD_NO'] = customer_df['COUNT_DEBITCARD'].max()
    diff_credit_list = customer_df['COUNT_CREDITCARD'].diff().tolist()
    diff_credit_list[0] = 0.0
    diff_debit_list = customer_df['COUNT_DEBITCARD'].diff().tolist()
    diff_debit_list[0] = 0.0
    subscribe_credit = False
    unsubscribe_credit = False
    subscribe_debit = False
    unsubscribe_debit = False

    for j in range(len(diff_credit_list) - 1):
      if (diff_credit_list[j + 1] - diff_credit_list[j] > 0):
        subscribe_credit = True
      if (diff_credit_list[j + 1] - diff_credit_list[j] < 0):
        unsubscribe_credit = True

    for j in range(len(diff_debit_list) - 1):
      if (diff_debit_list[j + 1] - diff_debit_list[j] > 0):
        subscribe_debit = True
      if (diff_debit_list[j + 1] - diff_debit_list[j] < 0):
        unsubscribe_debit = True
    
    card_features['SUBSCRIBE_CREDITCARD'] = subscribe_credit
    card_features['UNSUBSCRIBE_CREDITCARD'] = unsubscribe_credit
    card_features['SUBSCRIBE_DEBITCARD'] = subscribe_debit
    card_features['UNSUBSCRIBE_DEBITCARD'] = unsubscribe_debit
    card_feature_result.append(card_features)
  df = pd.DataFrame(card_feature_result)
  return df 
