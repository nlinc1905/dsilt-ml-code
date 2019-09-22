from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

class HClust():

    def __init__(self, input_data, linkage_method,
                 distance_metric, data_labels):
        self.input_data = input_data
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.data_labels = data_labels

    def cluster(self):
        self.hc_model = linkage(self.input_data,
                                method=self.linkage_method,
                                metric=self.distance_metric)

    def plot_dendrogram(self):
        plt.figure(figsize=(12, 5))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Distance')
        plt.ylabel('Song')        
        dendrogram(self.hc_model,
                   orientation='left',
                   leaf_font_size=10.,  #Specify font size of x-axis labels
                   labels=self.data_labels
                   )
        plt.tight_layout()
        plt.show()
