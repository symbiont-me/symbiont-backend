#!/bin/sh
# Check if the MongoDB container named "symbiont-test" exists
if [ "$(docker ps -aq -f name=symbiont-test)" ]; then
    echo "Starting existing container 'symbiont-test'."
    docker start symbiont-test
else
    # Run MongoDB container in detached mode
    echo "Creating and starting new 'symbiont-test' container."
    docker run -d --name symbiont-test -p 27017:27017 mongo
fi

# Check if the Qdrant container named "symbiont-qdrant-test" exists
if [ "$(docker ps -aq -f name=symbiont-qdrant-test)" ]; then
    echo "Starting existing container 'symbiont-qdrant-test'."
    docker start symbiont-qdrant-test
else
    # Run Qdrant container in detached mode
    echo "Creating and starting new 'symbiont-qdrant-test' container."
    docker run -d --name symbiont-qdrant-test -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
fi

# Wait for a moment to ensure containers are fully up
sleep 5  # Adjust sleep time if needed for your containers to fully initialize

# Run symbiont application after Docker containers are up
echo "Starting symbiont application..."
poetry run uvicorn symbiont.main:app --reload
