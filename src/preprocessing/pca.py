from sklearn.decomposition import PCA
import numpy as np

def apply_pca(data, n_components=2):
    """
    Apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset.

    Parameters:
    - data: numpy array or pandas DataFrame, the input data to be transformed.
    - n_components: int, the number of principal components to keep.

    Returns:
    - transformed_data: numpy array, the data transformed into the PCA space.
    - pca: PCA object, the fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca

def explained_variance_ratio(pca):
    """
    Get the explained variance ratio of each principal component.

    Parameters:
    - pca: PCA object, the fitted PCA model.

    Returns:
    - variance_ratio: numpy array, the explained variance ratio of each component.
    """
    return pca.explained_variance_ratio_