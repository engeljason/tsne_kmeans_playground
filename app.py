import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

filename = 'crypto_data.csv'
original_df = pd.read_csv(filename, index_col=0)
df = original_df[original_df['IsTrading']].drop(columns=['IsTrading', 'CoinName'])
df.dropna(inplace=True)
df = df[df['TotalCoinsMined'] > 0]
df['TotalCoinSupply'] = df['TotalCoinSupply'].astype('float64')
dum_df = pd.get_dummies(df, drop_first=True)


t_list = list(range(10, 110, 10))+list(range(100, 1100, 100))
for t in t_list:
    scaler = StandardScaler()
    pca = PCA(n_components=0.9, random_state=44312)
    tsne = TSNE(random_state=44312, learning_rate=t)

    scaled_data = scaler.fit_transform(dum_df)
    pca_data = pca.fit_transform(scaled_data)
    tsne_data = tsne.fit_transform(pca_data)

    x, y = zip(*tsne_data)
    
    plt.scatter(x, y)
    path = f'Graphs/rate_{t}'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/cluster_plot_{t}.png')
    plt.clf()

    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_jobs=3, random_state=44312)
        kmeans.fit_transform(tsne_data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1,11),inertias)
    plt.xticks(range(1,11))
    plt.savefig(f'{path}/k_means_analysis.png')
    plt.clf()
