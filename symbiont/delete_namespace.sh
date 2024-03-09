#!/bin/bash

# Define path to the Python script
script_path="symbiont/src/scripts/delete_pinecone_vecs.py"
# Use poetry to run the script
poetry run python3 $script_path 

