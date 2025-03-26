import unittest
from src.visualization.scatter_plot import plot_clusters
from src.visualization.elbow_curve import plot_elbow_curve
from src.visualization.dendrogram import plot_dendrogram

class TestVisualization(unittest.TestCase):

    def test_plot_clusters(self):
        # Test the scatter plot function with sample data
        sample_data = [[1, 2], [1, 4], [1, 0],
                       [4, 2], [4, 4], [4, 0]]
        sample_labels = [0, 0, 0, 1, 1, 1]
        try:
            plot_clusters(sample_data, sample_labels)
            result = True  # If no exception is raised, the plot function works
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_clusters function failed to execute without error.")

    def test_plot_elbow_curve(self):
        # Test the elbow curve function with sample data
        sample_inertia = [10, 8, 6, 4, 2]
        try:
            plot_elbow_curve(sample_inertia)
            result = True  # If no exception is raised, the plot function works
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_elbow_curve function failed to execute without error.")

    def test_plot_dendrogram(self):
        # Test the dendrogram function with sample data
        sample_linkage_matrix = [[0, 1, 1, 2], [2, 3, 1, 4]]  # Example linkage matrix
        try:
            plot_dendrogram(sample_linkage_matrix)
            result = True  # If no exception is raised, the plot function works
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_dendrogram function failed to execute without error.")

if __name__ == '__main__':
    unittest.main()