#!/bin/bash

# Define your domain
DOMAIN="http://127.0.0.1:8000/add-resource-to-db"

# Set file path and MIME type
FILE_PATH="/Users/leviathan/Documents/2103.08712v2.pdf"
FILE_TYPE="mimeType" # Replace with your file's MIME type

# StudyResource fields
STUDY_ID="Pp0ZYO6EL54A2XiIr7Eu"

# Construct JSON payload
RESOURCE_JSON=$(cat <<EOF
{
  "studyId": "$STUDY_ID",
}
EOF
)

# Execute curl command
curl -X POST "$DOMAIN" \
     -F "file=@$FILE_PATH;type=$FILE_TYPE" \
     -F "resource=$RESOURCE_JSON;type=application/json"

# End of script
