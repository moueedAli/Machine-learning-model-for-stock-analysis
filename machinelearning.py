import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
import plotly.graph_objects as go

from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


ticker = "TSLA"
df = yf.download(ticker, start="2000-01-01", end=date.today())
df['Date'] = df.index  
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].round(2)

df['Date'] = pd.to_datetime(df['Date'])
df['day'] = df['Date'].dt.day.astype('Int64')
df['month'] = df['Date'].dt.month.astype('Int64')
df['year'] = df['Date'].dt.year.astype('Int64')

# relevant for QRapporter 
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)

df.to_csv('${ticker}.csv', index=False)  

df.to_csv('${ticker}.csv', index=False)
df = pd.read_csv('${ticker}.csv')
df = df[df['Close'] != ticker]


cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
# print(df.head(21))
# print(df.isnull().sum()) # sjekker for null verdier i dataen

# print(df.to_string()) printer ut all data, ikke alltid nødvendig

# lager en graf med data fra csv for å se trenden til aksjen gitt data sett
plt.figure(figsize=(15,5))
plt.plot(df['Close'], label='Close Price')
plt.title('Meta Close Price (2024)', fontsize=15)
plt.ylabel('Price in dollars')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
#  plt.show()  

# lager serie med histogrammer for de ulike funksjonene gitt under
# viser fordeling av ulike verdier, hjelper med valg av riktig preprocessing
# kort sagt gir det intuisjon om datastrukturen, avgjørende før man bygger modeller eller trekker konklusjoner
# viser i form av histogrammer/diagrammer
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.figure(figsize=(20,10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribusjon av {col}')
    
plt.tight_layout()
#  plt.show()


# viser samme data som tidligere i form av box/blokker, ofte enklere å visualisere outliers
# outliers er ekstreme verdier i datasettet. Skiller seg kraftig ut fra majoriteten
# kommer av spesielel hendelse (nyheter, QRapporter, panikk i marked), feil i data eller naturlig variasjon
# disse er viktig å fange da de kan gi forstyrrelser i modeller, eller være verdifulle signaler

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.boxplot(df[col])
    plt.title(f'Distribusjon av {col}')
    
plt.tight_layout()
#  plt.show()  

# Feature engineering er prosessen hvor man lager, forbedrer eller trasnformerer variabler (features)
# Gjør dataen enda mer nyttig for analysen

df['PriceChange'] = df['Close'] - df['Open']
df['Direction'] = np.where(df['PriceChange'] > 0, 1, 0)

# akjseprisens svingning ila dagen
df['Volatility'] = df['High'] - df['Low']

# prosentvis endring i pris fra dagen før
df['PctChange'] = df['Close'].pct_change()

# gjennomsnittlig pris av akjsen i en gitt tidsperiode, trendindikator
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# enda mer data å mate modellen 
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# beregner RSI, hvor sterkt aksjen har beveger seg opp eller ned de siste dagene, vindu på 14 dager
# > 70 => aksjen er overkjøpt, mulig nedgang i vente 
# < 70 => akjsen er underkjøpt, mulig oppgang i vente  
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = compute_rsi(df)

# beregner MACD, forskjellen mellom to SMA, MACD linje er forskjellen, signal linje er SMA av MACD, ofte 9 dager
# MACD krysser over signal, bullish signal. MACD krysser under signal, bearish signal 
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

df['MACD'], df['MACD_signal'] = compute_macd(df)

# beregner bollinger bandwidth, brukes for å analysere prisendring/volatilitet og potensielle pris reverseringer
# gjennomsnittet av prisen til aksjen de siste 20 tidsperiodene
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# regner ut de siste 20 tidsperiodenes standardavvik
df['SD'] = df['Close'].rolling(window=20).std()

df['UB'] = df['SMA_20'] + 2 * df['SD']
df['LB'] = df['SMA_20'] - 2 * df['SD']

plot_df = df.dropna(subset=['SMA_20', 'UB', 'LB', 'Close']).copy()

# Create a Plotly figure
fig = go.Figure()

fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['UB'], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['LB'], mode='lines', name='Lower Bollinger Band', fill='tonexty', line=dict(color='green')))
fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA_20'], mode='lines', name='Middle Bollinger Band', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], mode='lines', name='Price', line=dict(color='black')))

fig.update_layout(title=ticker + ' Stock Price with Bollinger Bands', xaxis_title='Date', yaxis_title='Price', showlegend=True)

# når prisen beveger seg nære UB, kan det signalisere overkjøpte forhold, som kan gi en potensiell nedgang
# på samme måte når prisen beveger seg nære og rundt LB, kan det signalisere at akjsen er oversolgt, og en potensiell oppgang 
fig.show()

# beregning av VWAP 
df['VWAP'] = (((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']).cumsum() / df['Volume'].cumsum()

# Check for crossovers in the last two rows
last_row = df.iloc[-1]
second_last_row = df.iloc[-2]

if second_last_row['Close'] > second_last_row['VWAP'] and last_row['Close'] < last_row['VWAP']:
    print('Price Cross Below VWAP')
elif second_last_row['Close'] < second_last_row['VWAP'] and last_row['Close'] > last_row['VWAP']:
    print('Price Cross Above VWAP')
else:
    print('No Crossover')

# print(df[['Open', 'Close', 'PriceChange', 'Direction', 'Volatility']].head(21)) 

# visuell innsikt i hvordan de nye features er fordelt, gir tegn på skjevhet, outlies eller om de er balanserte
features1 = ['PriceChange', 'Volatility', 'PctChange']
plt.figure(figsize=(20,10))

for i, col in enumerate(features1):
    plt.subplot(2, 2, i+1)
    sb.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribusjon av {col}')

plt.tight_layout()
#  plt.show() 

# gir et diagram av gjennomsnittsprisen til aksjen for de ulike variablene under 
# ser at prisen har jevnt økt fra start. Liten nedgang fra 15 til 16, men jevn økning deretter
# en dobling fra 19 til 20 observert
data_grouped = df.groupby('year')[['Open', 'High', 'Low', 'Close']].mean()

plt.figure(figsize=(20,10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i+1)
    plt.bar(data_grouped.index, data_grouped[col], color='skyblue')
    plt.title(f'Årlig gjennomsnitt av {col}')
    plt.xlabel('År')
    plt.ylabel('Pris (USD)')
    plt.grid(True)

plt.tight_layout()
#  plt.show()
 
 
# fra dataen gruppert over kan vi gjøre nyttige observasjoner
# ved kvartalsmåned er prisen på akjsen generelt høyere, men volumet er lavere sammenlignet med ikke kvartalsmåned 
df.drop(['Date', 'day', 'month', 'year'], axis=1).groupby('is_quarter_end').mean()

# plt.close('all') # lukker alle diagrammer

# gir et pai diagram med 0 eller 1. 1 betyr at aksjen stiger neste dag, 0 at akjsen faller eller står stille
plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
#  plt.show()

# Velg features og target
features = ['open-close', 'low-high', 'Volatility', 'PctChange', 'SMA_50', 'RSI', 'MACD', 'MACD_signal']
df_model = df[features + ['target']].dropna()

X = df_model[features]
y = df_model['target']

# Splitt data
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Skaler data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tren modell
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluer
y_pred = model.predict(X_test_scaled)
print('Todays stock: ' + ticker)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# [[TN, FP]
#   [FN TP]]
# TN = True negative, modellen sa nedgang og det var riktig
# FP = False positive, modellen sa oppgang, men det var nedang
# FN = False negative, modellen sa nedgang, men det var oppgang
# TP = True positive, modellen sa oppgang, og det var riktig  
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print(df['target'].value_counts(normalize=True))

latest = df[features].tail(1).fillna(0)
prediction = model.predict(scaler.transform(latest))[0]
