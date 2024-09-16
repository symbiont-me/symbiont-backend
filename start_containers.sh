#!/bin/bash

# Set the MONGODB_VERSION environment variable if it's not already set
export MONGODB_VERSION=${MONGODB_VERSION:-latest}

# Function to clean and remove Docker containers
clean_container() {
  local container_name=$1
  if [ "$(docker ps -q -f name=$container_name)" ]; then
    echo "Stopping container $container_name..."
    docker stop $container_name
  fi
  if [ "$(docker ps -aq -f name=$container_name)" ]; then
    echo "Removing container $container_name..."
    docker rm $container_name
  fi
}

# Function to clean data directory
clean_data_dir() {
  local data_dir=$1
  if [ -d "$data_dir" ]; then
    echo "Cleaning data directory $data_dir..."
    rm -rf "$data_dir"
  fi
  mkdir -p "$data_dir"
}

# Clean and remove the specified containers
clean_container symbiont-local-mongodb
clean_container symbiont-local-qdrant
clean_container symbiont-local-supertokens

# Clean data directories
clean_data_dir "$(pwd)/data"                              # MongoDB
clean_data_dir "$(pwd)/qdrant_storage"                    # Qdrant

# Run the new instances of the containers with specific names
echo "Starting new instances of the containers as symbiont-local..."

docker run --name symbiont-local-mongodb -d -p 27017:27017 -v $(pwd)/data:/data/db mongodb/mongodb-community-server:$MONGODB_VERSION
docker run --name symbiont-local-qdrant -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
docker run --name symbiont-local-supertokens -d -p 3567:3567 registry.supertokens.io/supertokens/supertokens-mysql

echo "All containers are up and running as symbiont-local."
