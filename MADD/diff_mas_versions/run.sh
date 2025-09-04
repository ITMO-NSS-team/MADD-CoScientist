#!/bin/bash

source /root/miniconda3/bin/activate llm_pipeline
python multi_agents_system/utils/update_yamal_config.py
tmux new-session -d -s ollama_serve \; send-keys "ollama serve" Enter 
sleep 10
ollama pull llama3.1
tmux new-session -d -s llama3 \; send-keys "run llama3.1" Enter 
sleep 3
nohup python -u multi_agents_system/inference.py &> links.txt &
sleep 10
cat links.txt