def get_trans_feature(df):
  df["TRANS_DATE"] = pd.to_datetime(df["TRANS_DATE"], format="%Y-%m-%d")
  df["IB_REGISTER_DATE"] = pd.to_datetime(df["IB_REGISTER_DATE"], format="%Y-%m-%d")
  df["delta"] = (df["TRANS_DATE"] - df["IB_REGISTER_DATE"]).dt.days
  total_df = df.groupby(["CUSTOMER_NUMBER", "label"]
            ).agg({"TRANS_NO": "sum", "TRANS_AMOUNT": "sum"})
  total_df = total_df.reset_index().fillna(0)

  hour_df = df.groupby(["CUSTOMER_NUMBER", "TRANS_HOUR"]
            ).agg({"TRANS_NO": "sum", "TRANS_AMOUNT": "sum"}).unstack("TRANS_HOUR")
  hour_df = hour_df / day_num
  hour_df = hour_df.reset_index().fillna(0)
  
  dow_df = df.groupby(["CUSTOMER_NUMBER", "DAY_OF_WEEK"]).agg(
      {"TRANS_NO": "mean", "TRANS_AMOUNT": "mean"}).unstack("DAY_OF_WEEK")
  dow_df = dow_df.reset_index().fillna(0)
  
  day_df = df.groupby(["CUSTOMER_NUMBER", "day"]).agg(
      {"TRANS_NO": "sum", "TRANS_AMOUNT": "sum"}).unstack("day")
  day_df = day_df / day_num
  day_df = day_df.reset_index().fillna(0)
  
  d_lv1_df = df.groupby(["CUSTOMER_NUMBER", "TRANS_LV1"]
                      ).agg({"TRANS_NO": "sum", "TRANS_AMOUNT": "sum"}).unstack("TRANS_LV1")
  d_lv1_df = d_lv1_df / day_num
  d_lv1_df = d_lv1_df.reset_index().fillna(0)

  d_lv2_df = df.groupby(["CUSTOMER_NUMBER", "TRANS_LV2"]
                      ).agg({"TRANS_NO": "sum", "TRANS_AMOUNT": "sum"}).unstack("TRANS_LV2")
  d_lv2_df = d_lv2_df / day_num
  d_lv2_df = d_lv2_df.reset_index().fillna(0)

  result = pd.merge(left=hour_df, right=total_df, on="CUSTOMER_NUMBER", how="left")
  result = pd.merge(left=result, right=dow_df, on="CUSTOMER_NUMBER", how="left")
  result = pd.merge(left=result, right=day_df, on="CUSTOMER_NUMBER", how="left")
  result = pd.merge(left=result, right=d_lv1_df, on="CUSTOMER_NUMBER", how="left")
  result = pd.merge(left=result, right=d_lv2_df, on="CUSTOMER_NUMBER", how="left")
  
  return result
