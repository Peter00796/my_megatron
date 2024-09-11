import matplotlib.pyplot as plt
import numpy as np
import torch
import os

import sys
sys.path.append(os.path.abspath(os.path.join('..')))

# Load the checkpoint
checkpoint_path = '/mnt/pengyanxin/my_megatron/examples/gpt3/gpt-345m-ckpts/iter_0000030/mp_rank_00/model_optim_rng.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract the tensor you want to visualize
tensor = checkpoint['optimizer']['optimizer']['state'][0]['exp_avg'].flatten().numpy()

# Sample a subset of the data
sample_size = min(10000, len(tensor))
indices = np.random.choice(len(tensor), sample_size, replace=False)
sampled_tensor = tensor[indices]
sampled_indices = np.arange(len(tensor))[indices]

# Set the value limit for the main plot
value_limit = 2.5e-5  # Adjust this value to change the y-axis range

# Create the main scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(sampled_indices, sampled_tensor, alpha=0.5, s=1)
plt.ylim(-value_limit, value_limit)
plt.title(f'Distribution of Optimizer States (|value| <= {value_limit})')
plt.xlabel('Index')
plt.ylabel('Value')

# Add text to indicate outliers
outliers = np.sum(np.abs(sampled_tensor) > value_limit)
plt.text(0.95, 0.95, f'Outliers: {outliers}', 
         horizontalalignment='right', verticalalignment='top', 
         transform=plt.gca().transAxes, fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()

# Save the plot
output_filename = os.path.join('pictures', 'optimizer_state_distribution_focused.png')
os.makedirs('pictures', exist_ok=True)
plt.savefig(output_filename, dpi=300)
print(f"Plot saved as {output_filename}")

plt.show()