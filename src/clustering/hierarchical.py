def hierarchical_clustering(data, method='ward', metric='euclidean'):
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt

    # Perform hierarchical clustering
    Z = linkage(data, method=method, metric=metric)

    return Z

def plot_dendrogram(Z, labels=None):
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

def fit_hierarchical_clustering(data, method='ward', metric='euclidean', plot=True):
    Z = hierarchical_clustering(data, method, metric)
    
    if plot:
        plot_dendrogram(Z)

    return Z