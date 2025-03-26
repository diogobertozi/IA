import pytest
import numpy as np
from src.preprocessing.normalization import normalize_data
from src.preprocessing.pca import perform_pca

def test_normalize_data():
    # Test normalization of a simple dataset
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    normalized_data = normalize_data(data)
    
    # Check if the normalized data has mean close to 0 and std close to 1
    assert np.allclose(np.mean(normalized_data, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(normalized_data, axis=0), 1, atol=1e-7)

def test_perform_pca():
    # Test PCA on a simple dataset
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pca_result = perform_pca(data, n_components=2)
    
    # Check if the PCA result has the correct shape
    assert pca_result.shape[1] == 2
    assert pca_result.shape[0] == data.shape[0]