#!/bin/bash

# List of model strings
models=("Huginn-13B-v4-GPTQ" "Huginn-v3-13B-GPTQ" "Huginn-13B-v4.5-GPTQ", "LoKuS-13B-GPTQ")  # Add more models as required

# List of ports
ports=(60113, 60180, 60151, 60181)  # Add more ports as required

# Ensure that the number of models and ports are the same
if [[ ${#models[@]} -ne ${#ports[@]} ]]; then
    echo "Error: The number of models and ports must be the same."
    exit 1
fi

# Start PM2 processes
for i in "${!models[@]}"; do
    model=${models[$i]}
    port=${ports[$i]}
    gpu_id=$i  # This assumes you start with GPU ID 0 and increase by 1 for each model

    echo "Starting model $model on port $port with GPU ID $gpu_id..."
    pm2 start host_gptq.py --name "$model" --interpreter python3 -- "$model" "$port" "$gpu_id"
done

echo "All models started!"
