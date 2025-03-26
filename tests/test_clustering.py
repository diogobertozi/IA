import unittest
from src.clustering.kmeans import KMeans
from src.clustering.hierarchical import HierarchicalClustering
from src.preprocessing.normalization import normalize_data
from src.preprocessing.pca import apply_pca

class TestClusteringAlgorithms(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = [[1, 2], [1, 4], [1, 0],
                     [4, 2], [4, 4], [4, 0]]
        self.normalized_data = normalize_data(self.data)
        self.pca_data = apply_pca(self.normalized_data)

    def test_kmeans(self):
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(self.pca_data)
        self.assertEqual(len(kmeans.predict(self.pca_data)), len(self.pca_data))

    def test_hierarchical_clustering(self):
        hierarchical = HierarchicalClustering()
        hierarchical.fit(self.pca_data)
        self.assertIsNotNone(hierarchical.get_dendrogram())

if __name__ == '__main__':
    unittest.main()