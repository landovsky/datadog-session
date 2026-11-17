#!/bin/bash

# Force recreate only the swagbot container
docker compose up -d --force-recreate swagbot

# Wait for Flask to be ready before returning
echo "Waiting for SwagBot to be ready..."
until curl -sf http://localhost:3000/ > /dev/null 2>&1; do
    sleep 1
done
sleep 3  # Allow Vertex AI client and LangGraph workflow to fully initialize
echo "✅ SwagBot is ready!"