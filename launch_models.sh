#!/bin/bash

# List of model strings
models=("Huginn-13B-v4-GPTQ" "Huginn-v3-13B-GPTQ" "Huginn-13B-v4.5-GPTQ" "LoKuS-13B-GPTQ")  # Add more models as required

# List of ports
ports=(60113 60180 60151 60181)  # Add more ports as required

# Ensure that the number of models and ports are the same
if [[ ${#models[@]} -ne ${#ports[@]} ]]; then
    echo "Error: The number of models and ports must be the same."
    exit 1
fi

# Get public IP
ip=$(curl -s https://ifconfig.me)

# Initialize an empty array for URLs
declare -a urls

# Start PM2 processes
for i in "${!models[@]}"; do
    model="../${models[$i]}"  # The model path is relative to the parent directory
    port=${ports[$i]}
    gpu_id=$i  # This assumes you start with GPU ID 0 and increase by 1 for each model

    # Append the URL to the urls array
    urls+=("http://$ip:$port")

    echo "Starting model $model on port $port with GPU ID $gpu_id..."
    pm2 start host_gptq.py --name "${models[$i]}" --interpreter python3 -- "$model" "$port" "$gpu_id"
done

# Print the list of URLs as a Python list
echo "All models started!"
echo -n "["
for i in "${!urls[@]}"; do
    if [[ $i -ne 0 ]]; then
        echo -n ","
    fi
    echo -n "\"${urls[$i]}\""
done
echo "]"
