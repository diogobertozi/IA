def calculate_silhouette_score(X, labels):
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, labels)

def calculate_davies_bouldin_score(X, labels):
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(X, labels)

def calculate_inertia(X, kmeans_model):
    return kmeans_model.inertia_

def get_optimal_k_elbow_method(X, max_k):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    inertia = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertia.append(calculate_inertia(X, kmeans))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values)
    plt.grid()
    plt.show()

def standardize_data(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)