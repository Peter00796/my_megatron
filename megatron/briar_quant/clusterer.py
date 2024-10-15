import torch
import numpy as np
from scipy import stats
import cupy as cp
import time

class Clusterer:


    @classmethod
    def deterministic_clusters(cls, data, num_clusters):
        # Ensure data is on the correct device
        device = data.device
        # Flatten data
        data = data.flatten()
        # Initialize cluster centers deterministically
        min_val, max_val = data.min(), data.max()
        cluster_centers = torch.linspace(min_val, max_val, steps=num_clusters, device=device)
        
        # Perform a fixed number of K-Means iterations
        for _ in range(10):  # Fixed number of iterations
            # Assign points to nearest cluster center
            distances = torch.abs(data.unsqueeze(1) - cluster_centers.unsqueeze(0))
            cluster_assignments = distances.argmin(dim=1)
            # Update cluster centers
            for i in range(num_clusters):
                mask = cluster_assignments == i
                if mask.any():
                    cluster_centers[i] = data[mask].mean()
                else:
                    # Handle empty clusters if necessary
                    pass
        # Compute quantization thresholds
        thresholds = (cluster_centers[:-1] + cluster_centers[1:]) / 2
        return cluster_centers, thresholds
    
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
        Return cluster centers, cluster labels, and original shape.
        """
        print(f"input_tensor shape: {input_tensor.shape}")
        original_shape = input_tensor.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        data = input_tensor.flatten()
        
        # Ensure interval_number is reasonable
        interval_number = min(interval_number, data.numel() // 100)  # At least 100 elements per cluster on average
        
        # Compute min and max values
        min_val, max_val = data.min(), data.max()
        
        # Create evenly spaced cluster centers
        cluster_centers = torch.linspace(min_val, max_val, steps=interval_number, device=device)
        
        # Assign data to clusters
        distances = torch.abs(data.unsqueeze(1) - cluster_centers.unsqueeze(0))
        cluster_labels = distances.argmin(dim=1)

        
        # Update cluster centers based on assigned data
        for i in range(interval_number):
            mask = cluster_labels == i
            if mask.any():
                cluster_centers[i] = data[mask].mean()
        
        return cluster_centers, cluster_labels, original_shape
            
        
        
        

    
    @classmethod
    def cluster_optimizer_states(cls, optimizer_states, interval_number):
        """
        This function cluster the optimzier states and store cluster lebles and original shapes
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clustered_states = {}
        cluster_labels_states = {}
        original_shapes = {}
        # initialize the clustered_states, cluster_labels_states, original_shapes   
        for key in optimizer_states:
            optimizer_state_dict = optimizer_states[key]
            clustered_state = {}
            cluster_labels_state = {}
            shape_dict = {}
            for state_key in optimizer_state_dict:
                print(f"state_key: {state_key}")   
                print(f"optimizer_state_dict[state_key]: {optimizer_state_dict[state_key]}")
                tensor = optimizer_state_dict[state_key].to(device)
                
                clusters, cluster_labels, original_shape = cls.naive_normal_clusters(tensor, interval_number)
                clustered_state[state_key] = clusters
                cluster_labels_state[state_key] = cluster_labels.to(torch.uint8)
                shape_dict[state_key] = original_shape
            clustered_states[key] = clustered_state
            cluster_labels_states[key] = cluster_labels_state
            original_shapes[key] = shape_dict
        return clustered_states, cluster_labels_states, original_shapes
    


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
    def de_cluster_optimizer_states(cls, clustered_states, cluster_labels_states, original_shapes):
        """
        De-clusters the tensors in the optimizer states
        """
        de_clustered_states = {}
        for key in clustered_states:
            clustered_state_dict = clustered_states[key]
            cluster_labels_state = cluster_labels_states[key]
            de_clustered_state = {}
            for state_key, clusters in clustered_state_dict.items():
                cluster_labels = cluster_labels_state[state_key]
                original_shape = original_shapes[key][state_key]
                
                data = torch.empty(cluster_labels.numel(), device=cluster_labels.device)
                for i, cluster in enumerate(clusters):
                    if cluster.numel() > 0:
                        mask = cluster_labels == i
                        data[mask] = cluster
                de_clustered_tensor = data.view(original_shape)
                de_clustered_state[state_key] = de_clustered_tensor
            de_clustered_states[key] = de_clustered_state
        return de_clustered_states

