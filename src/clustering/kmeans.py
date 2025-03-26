def fit_kmeans(data, n_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans

def predict_clusters(kmeans, data):
    return kmeans.predict(data)

def calculate_inertia(data, max_clusters):
    inertia = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = fit_kmeans(data, n_clusters)
        inertia.append(kmeans.inertia_)
    return inertia

def elbow_method(inertia):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(inertia) + 1), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()