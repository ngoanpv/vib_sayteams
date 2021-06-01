import pandas as pd, numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py #visualization
py.init_notebook_mode(connected=True) #visualization
import plotly.graph_objs as go #visualization
import plotly.tools as tls #visualization
import plotly.figure_factory as ff #visualization
import datetime

customer = pd.read_csv('/content/drive/Shareddrives/vib_challenge/data/Track 1 Datarathon/1.Data_Customer.csv')
activity = pd.read_csv('/content/drive/Shareddrives/vib_challenge/data/Track 1 Datarathon/3.Data_MyVIB_Activity.csv')

features = df[['CUSTOMER_NUMBER', 'IB_REGISTER_DATE']]

list_cn = list(features['CUSTOMER_NUMBER'])
activity = activity[activity['CUSTOMER_NUMBER'].isin(list_cn)].reset_index(drop = True)
activity = pd.merge(activity, features, how='left', on='CUSTOMER_NUMBER')
activity['ACTIVITY_DATE'] = pd.to_datetime(activity['ACTIVITY_DATE'])
activity['IB_REGISTER_DATE'] = pd.to_datetime(activity['IB_REGISTER_DATE'])
activity['OBSERVE_TIME'] = activity['IB_REGISTER_DATE'].apply(lambda r: r + datetime.timedelta(days=30))
activity['IS_CHECK'] = activity['ACTIVITY_DATE'] <= activity['OBSERVE_TIME']

temp = activity.groupby(['CUSTOMER_NUMBER', 'IS_CHECK'])['ACTIVITY_HOUR'].count().reset_index()
temp = temp[temp['IS_CHECK']][['CUSTOMER_NUMBER', 'ACTIVITY_HOUR']]
temp.rename(columns={'ACTIVITY_HOUR':'total_active_hour'}, inplace=True)
features = features.merge(temp, how='left', on='CUSTOMER_NUMBER')

temp = activity.groupby(['CUSTOMER_NUMBER', 'IS_CHECK'])['ACTIVITY_HOUR'].nunique().reset_index()
temp = temp[temp['IS_CHECK']][['CUSTOMER_NUMBER', 'ACTIVITY_HOUR']]
temp.rename(columns={'ACTIVITY_HOUR':'total_active_hour_uniq'}, inplace=True)
features = features.merge(temp, how='left', on='CUSTOMER_NUMBER')

total_nb_action = activity.groupby(['CUSTOMER_NUMBER', 'IS_CHECK'])['ACTIVITY_NO'].count().reset_index()
total_nb_action = total_nb_action[total_nb_action['IS_CHECK']][['CUSTOMER_NUMBER', 'ACTIVITY_NO']]
total_nb_action.rename(columns={'ACTIVITY_NO':'total_nb_action'}, inplace=True)
features = features.merge(total_nb_action, how='left', on='CUSTOMER_NUMBER')

total_nb_action = activity.groupby(['CUSTOMER_NUMBER', 'IS_CHECK'])['ACTIVITY_NO'].nunique().reset_index()
total_nb_action = total_nb_action[total_nb_action['IS_CHECK']][['CUSTOMER_NUMBER', 'ACTIVITY_NO']]
total_nb_action.rename(columns={'ACTIVITY_NO':'total_nb_action_uniq'}, inplace=True)
features = features.merge(total_nb_action, how='left', on='CUSTOMER_NUMBER')

features['avg_act_no'] = features['total_nb_action'] / 30

most_active_day = activity.groupby(['CUSTOMER_NUMBER', 'IS_CHECK'])['DAY_OF_WEEK'].agg(pd.Series.mode).reset_index()
most_active_day = most_active_day[most_active_day['IS_CHECK']][['CUSTOMER_NUMBER', 'DAY_OF_WEEK']]
most_active_day.rename(columns={'DAY_OF_WEEK':'most_active_day'}, inplace=True)
features = features.merge(most_active_day, how='left', on='CUSTOMER_NUMBER')

