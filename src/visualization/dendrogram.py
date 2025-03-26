def plot_dendrogram(linkage_matrix, labels=None):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

def save_dendrogram(linkage_matrix, filename, labels=None):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.savefig(filename)
    plt.close()