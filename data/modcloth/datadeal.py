import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('modcloth_raw.csv', names=['item_id','user_id','rating','timestamp','size','fit','user_attr','model_attr','category','brand','year','split'])

dataset = dataset.drop(columns = ['size','fit','user_attr','model_attr','category','brand','year','split'])
temp = dataset.loc[:, 'user_id']
dataset = dataset.drop(columns = ['user_id'])
dataset.insert(0, 'user_id', temp)

userId, itemId = ['user_id','item_id']

le = LabelEncoder()
dataset[userId] = le.fit_transform(dataset[userId])
dataset[itemId] = le.fit_transform(dataset[itemId])

print(str(max(dataset[userId])+1))
print(str(max(dataset[itemId])+1))

dataset.to_csv('ratings.csv', header=False, index=False)