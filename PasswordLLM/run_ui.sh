#!/bin/bash

echo "Starting Password Security Assistant..."
echo ""

if [ ! -d "vector_stores" ]; then
    echo "Vector stores not found!"
    echo "Initializing vector stores first..."
    python initialize_rag.py
    echo ""
fi

if ! pgrep -x "ollama" > /dev/null; then
    echo "Ollama doesn't appear to be running"
    echo "Start Ollama with: ollama serve"
    echo ""
fi


echo "Launching web interface..."
echo "The app will open in your browser"
echo ""
streamlit run app.py


