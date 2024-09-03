import torch
import numpy as np
from scipy import stats
import cupy as cp

class Clusterer:

    @staticmethod
    def pre_process_data(data):
        """
        Flatten the data if it is not already flattened (dim > 1)
        """
        if data.dim() > 1:
            data = data.flatten()
        return data
    
    @staticmethod
    def calculate_mean_and_stddev(data):
        """
        Calculate the mean and standard deviation of the data
        """
        mean = data.mean().item()
        stddev = data.std(unbiased=True).item()
        return mean, stddev

    @staticmethod
    def divide_into_intervals(num_intervals, mean, stddev):
        """
        Divide the data into intervals based on the number of intervals using CuPy
        """
        quantiles = cp.linspace(0, 1, num_intervals + 1)
        intervals = cp.asnumpy(cp.random.normal(mean, stddev, size=len(quantiles)))
        return torch.tensor(intervals)

    @staticmethod
    def assign_to_clusters(data, intervals):
        """
        Assign the data to clusters based on the intervals
        """
        clusters = []
        for i in range(len(intervals) - 1):
            cluster = data[(data >= intervals[i]) & (data < intervals[i+1])]
            clusters.append(cluster)
        return clusters

    @classmethod
    def naive_normal_clusters(cls, input_tensor, interval_number):
        """
        Cluster the input tensor based on the normal distribution
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        input_tensor = cls.pre_process_data(input_tensor)
        mean, std_dev = cls.calculate_mean_and_stddev(input_tensor)
        intervals = cls.divide_into_intervals(interval_number, mean, std_dev).to(device)
        clusters = cls.assign_to_clusters(input_tensor, intervals)
        return clusters
    
    @classmethod
    def cluster_optimizer_states(cls, optimizer_states, interval_number):
        """
        Input of this function is the optimizer states dictionary
        This function receives the optimizer states dictionary and clusters the tensors in the optimizer states
        This function returns the clustered states and the original shapes of the tensors
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clustered_states = {}
        original_shapes = {}
        for key in optimizer_states:
            optimizer_state_dict = optimizer_states[key]
            clustered_state = {}
            shape_dict = {}
            for state_key in optimizer_state_dict:
                tensor = optimizer_state_dict[state_key].to(device)
                clusters = cls.naive_normal_clusters(tensor, interval_number)
                clustered_state[state_key] = clusters
                shape_dict[state_key] = tensor.shape
            clustered_states[key] = clustered_state
            original_shapes[key] = shape_dict
        return clustered_states, original_shapes
    


    ##################################################################### DE CLUSTERING #####################################################################

    @staticmethod
    def flatten_clusters(clusters):
        """
        Flatten the clusters and concatenate them
        """
        return torch.cat(clusters)
    
    @staticmethod
    def reshape_tensor(flat_tensor, original_shape):
        """
        Reshape the flattened tensor to the original shape
        """
        return flat_tensor.view(original_shape)
    
    @classmethod
    def de_cluster_optimizer_states(cls, clustered_states, original_shapes):
        """
        Input of this function is the clustered states dictionary and the original shapes dictionary (both saved on disk)
        This function receives the clustered states dictionary and the original shapes dictionary and de-clusters the tensors in the optimizer states
        This function returns the de-clustered states
        """
        de_clustered_states = {}
        for key in clustered_states:
            clustered_state_dict = clustered_states[key]
            de_clustered_state = {}
            for state_key in clustered_state_dict:
                clusters = clustered_state_dict[state_key]
                flat_tensor = cls.flatten_clusters(clusters)
                original_shape = original_shapes[key][state_key]
                de_clustered_tensor = cls.reshape_tensor(flat_tensor, original_shape)
                de_clustered_state[state_key] = de_clustered_tensor
            de_clustered_states[key] = de_clustered_state
        return de_clustered_states