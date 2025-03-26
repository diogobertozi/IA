def plot_elbow_curve(data, max_clusters=10):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # Calculate the sum of squared distances for each number of clusters
    ssd = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        ssd.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), ssd, marker='o')
    plt.title('Elbow Curve')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances (SSD)')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid()
    plt.show()