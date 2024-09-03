#!/bin/bash

DIRECTORY="/mnt/pengyanxin/my_megatron/examples/gpt3/gpt2_345m"
INTERVAL=60  # Check every 60 seconds
MIN_CHECKPOINTS=10  # Minimum number of checkpoints to keep

remove_oldest_checkpoint() {
    while true; do
        # Find all checkpoint directories starting with 'iter_'
        checkpoint_dirs=($(find "$DIRECTORY" -maxdepth 1 -type d -name 'iter_*' | sort -V))
        echo "Found checkpoint directories: $checkpoint_dirs"
        # Count the number of checkpoint directories
        dir_count=${#checkpoint_dirs[@]}
        echo "Current checkpoint count: $dir_count"

        # If there are more than the minimum number of checkpoints, remove the oldest
        if [ "$dir_count" -gt "$MIN_CHECKPOINTS" ]; then
            # Calculate how many directories to remove
            remove_count=$((dir_count - MIN_CHECKPOINTS))
            echo "Checkpoints to remove: $remove_count"
            for (( i=0; i<$remove_count; i++ )); do
                oldest_dir=${checkpoint_dirs[i]}
                echo "Removing checkpoint: $oldest_dir"
                if rm -rf "$oldest_dir"; then
                    echo "Successfully removed: $oldest_dir"
                else
                    echo "Failed to remove: $oldest_dir"
                fi
            done
        else
            echo "No checkpoints to remove. Current count ($dir_count) is not greater than minimum ($MIN_CHECKPOINTS)."
        fi

        echo "Checkpoint cleanup completed. Sleeping for $INTERVAL seconds."
        sleep $INTERVAL
    done
}

# Run the function in the background
remove_oldest_checkpoint