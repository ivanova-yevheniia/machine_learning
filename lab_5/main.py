import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

df = pd.read_csv("WQ-R.csv", sep=";")

def elbow_method(df):
    distortions = []
    k_values = range(1, 15)
    for k in k_values:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)
    kn = KneeLocator(k_values, distortions, curve='convex', direction='decreasing')

    #plot
    plt.figure(figsize=(16, 8))
    plt.plot(k_values, distortions, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return kn.knee

def aver_silhouette(df):
    k_values = range(2, 15)
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(df)
        silhouette_scores.append(silhouette_score(df, labels))
    optimal_clusters = k_values[np.argmax(silhouette_scores)]

    #plot
    plt.plot(k_values, silhouette_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('The Average Silhouette Method showing the optimal k')
    plt.show()

    return optimal_clusters

def prediciton_strength(df):
    k_values = range(2, 15)
    prediction_strengths = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(df)

        prediction_strength = np.mean(np.equal.outer(labels, labels))
        prediction_strengths.append(prediction_strength)

    #plot
    plt.plot(k_values, prediction_strengths)
    plt.xlabel('Number of clusters')
    plt.ylabel('Prediction strength')
    plt.title('The Predicition Strength Method showing the optimal k')
    plt.show()

    return np.argmax(prediction_strengths) + 2


if __name__ == '__main__':
    '''Визначити та вивести кількість записів'''
    num_rows, num_columns = df.shape
    print("Number of rows:", num_rows, " Number of columns:", num_columns)

    '''Видалити атрибут quality.'''
    df = df.drop('quality', axis=1)

    '''Вивести атрибути, що залишилися. '''
    print(list(df.columns))

    '''Використовуючи функцію KMeans бібліотеки scikit-learn, виконати
    розбиття набору даних на кластери з випадковою початковою
    ініціалізацією і вивести координати центрів кластерів.
    Оптимальну кількість кластерів визначити на основі початкового
    набору даних трьома різними способами: 
    1) elbow method; 
    2) average silhouette method; 
    3) prediction strength method (див. п. 9.2.3 Determining the Number of 
    Clusters книжки Andriy Burkov. The Hundred-Page Machine Learning Book). '''
    k_1 = elbow_method(df)
    print('Optimal k ELBOW METHOD: ', k_1)
    kmeans1 = KMeans(n_clusters=k_1, init='random')
    kmeans1.fit(df)
    cluster_centers1 = kmeans1.cluster_centers_
    print('Cluster Center: ', cluster_centers1)

    k_2 = aver_silhouette(df)
    print('Optimal k AVERAGE SILHOUETTE: ', k_2)
    kmeans2 = KMeans(n_clusters=k_2, init='random')
    kmeans2.fit(df)
    cluster_centers2 = kmeans2.cluster_centers_
    print('Cluster Center: ', cluster_centers2)

    k_3 = prediciton_strength(df)
    print('Otimal k PREDICTION STRENGTH: ', k_3)
    kmeans3 = KMeans(n_clusters=k_3, init='random')
    kmeans3.fit(df)
    labels_k3 = kmeans3.fit_predict(df)
    k_3silhouette = silhouette_score(df, labels_k3)
    cluster_centers3 = kmeans3.cluster_centers_
    print('Cluster Center: ', cluster_centers3)

    '''За раніш обраної кількості кластерів багаторазово проведіть
    кластеризацію методом k-середніх, використовуючи для початкової
    ініціалізації метод k-means++. 
    Виберіть найкращий варіант кластеризації. Який кількісний критерій
    Ви обрали для відбору найкращої кластеризації?'''
    k = k_2
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
    kmeans.fit(df)
    best_inertia = kmeans.inertia_
    best_iteration = 1
    for i in range(2, 10):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
        kmeans.fit(df)
        inertia = kmeans.inertia_
        if inertia < best_inertia:
            best_inertia = inertia
            best_iteration = i
    print('Найкраща кластеризація на ітерації: ', best_iteration, 'з критерієм inertia: ', best_inertia)


    '''Використовуючи функцію AgglomerativeClustering бібліотеки scikit-learn, 
    виконати розбиття набору даних на кластери. Кількість кластерів
    обрати такою ж самою, як і в попередньому методі. Вивести координати центрів кластерів. '''
    clustering = AgglomerativeClustering(n_clusters=k)
    clustering.fit(df)
    labels = clustering.labels_
    k_4silhouette = silhouette_score(df, labels)
    centroids = np.array([np.mean(df[labels == cluster_label], axis=0) for cluster_label in range(k)])
    print('AgglomerativeClustering: ', centroids)

    ''' Порівняти результати двох використаних методів кластеризації за silhouette_score'''
    print('KMeans: ', k_3silhouette)
    print('AgglomerativeClustering: ', silhouette_score(df, labels))









