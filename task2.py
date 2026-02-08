import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor
# print(sb.__version__)
import kagglehub
kagglepath=kagglehub.dataset_download("nelgiriyewithana/most-streamed-spotify-songs-2024")
print("DATASET PATH IS:",kagglepath)
files=os.listdir(kagglepath)
print('FILES IN THE PATH',files)
df=pd.read_csv(os.path.join(kagglepath,"Most Streamed Spotify Songs 2024.csv"),encoding='latin1')
df.dtypes
df['Shazam Counts']=(df['Shazam Counts'].astype(str).str.replace(',','',regex=False).astype(float).fillna(0))
print(df['Shazam Counts'].dtype)
df[['Soundcloud Streams','Pandora Track Stations','All Time Rank','Spotify Streams','Spotify Playlist Count','Spotify Playlist Reach','YouTube Views','YouTube Likes','TikTok Posts','TikTok Likes','TikTok Views','YouTube Playlist Reach','AirPlay Spins','SiriusXM Spins','Deezer Playlist Reach']]=(df[['Soundcloud Streams','Pandora Track Stations','All Time Rank','Spotify Streams','Spotify Playlist Count','Spotify Playlist Reach','YouTube Views','YouTube Likes','TikTok Posts','TikTok Likes','TikTok Views','YouTube Playlist Reach','AirPlay Spins','SiriusXM Spins','Deezer Playlist Reach']].astype(str).replace(',','',regex=True).astype(float).fillna(0))
print(df[['Soundcloud Streams','Pandora Track Stations','All Time Rank','Spotify Streams','Spotify Playlist Count','Spotify Playlist Reach','YouTube Views','YouTube Likes','TikTok Posts','TikTok Likes','TikTok Views','YouTube Playlist Reach','AirPlay Spins','SiriusXM Spins','Deezer Playlist Reach']])
df=df.drop(columns=['TIDAL Popularity'])
df['Pandora Streams']=(df['Pandora Streams'].astype(str).replace(',','',regex=True).astype(float).fillna(0))
print(df['Pandora Streams'])
df['Release Date']=pd.to_datetime(df['Release Date'])
#####################################################
'HISTOGRAM'
#####################################################
plt.figure(figsize=(10,6))
sb.histplot(df['Spotify Streams'],bins=50,kde=True)#kde is to represent the curve
# plt.xscale('log')
plt.title("Spotify Streams")
plt.xlabel('Streams')
plt.ylabel('Songs')
plt.show()
######################################################
'HEATMAP'
######################################################
plt.figure(figsize=(20,10))
sb.heatmap(df.corr(numeric_only=True),fmt=".3f",linewidths=0.5,annot=True,cmap='coolwarm')
plt.title("Correlation Matrix of Numerical Features")
plt.show()
######################################################
'SCATTERPLOT'
######################################################
plt.figure(figsize=(20,10))
plt.title("Spotify Streams vs Track Score")
sb.scatterplot(data=df,x='Spotify Streams',y='Track Score')
plt.show()
######################################################
'LINEGRAPH(ALL SONGS)'
######################################################
monthly_avg = df.groupby("Release Date")["Spotify Streams"].mean().reset_index()
plt.figure(figsize=(20,10))
sb.lineplot(data=df,x='Release Date',y='Spotify Streams',marker='o')
plt.title('Song Streams VS Release Date')
plt.xlabel('Spotify Streams')
plt.ylabel('Release Date')
plt.grid(True)
plt.show()
#######################################################
"LINEGRAPH(2024 SONGS)"
#######################################################
df_2024 = df[df['Release Date'].dt.year == 2024]
monthly_avg_2024 = (
    df_2024
    .groupby('Release Date')['Spotify Streams']
    .mean()
    .reset_index()
    # .sort_values('Release Date')
)
plt.figure(figsize=(20,10))
sb.lineplot(
    data=monthly_avg_2024,
    x='Release Date',
    y='Spotify Streams',
    marker='o'
)
plt.title("Average Streams by Release Month (2024)")
plt.xlabel("Release Date")
plt.ylabel("Spotify Streams")
plt.grid(True)
plt.show()
