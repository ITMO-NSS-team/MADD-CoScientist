## Agent-based LLM system

## Description
The multi-agent system consists of 4 agents. 
The main agent is the Conductor, an agent responsible for defining a tool to solve the user's question, it has tools available to it - call generative models, call functions to define properties of molecules, call Chat agent to answer general questions.

What can an agent system do?
-
- tell about its functionality, advise on its use, if necessary
- generate new molecules of the general spectrum and for specific cases - for example, a candidate molecule for Alzheimer's disease or Multiple Sclerosis.
- Determine the properties of molecules
- visualize molecules

Supported LLM models:
-

For the conductor agent (and others except Decomposer):
 - meta-llama/llama-3.1-70b-instruct
 - meta-llama/llama-3.1-8b-instruct
 - meta-llama/llama-3.2-90b-instruct
 - openai/o1-mini

 For the Decomposer agent:
 - llama-3.1-8b-q4


## Installation
- fill out config.yaml. It is mandatory to specify API KEY from personal account LLM service providers, address for generative models.
- bend the repository, go to the directory multi_agents_system
- create a container using DockerFile
```
docker build -t agents_system .
```
- start the container without specifying the required graphics cards as follows:
```
docker run --name agents_system -it --init agents_system bash

docker start agents_system
```

You can specify the required gpu's in the following way (instead of the 5th video card, list the numbers of the required video cards separated by commas):
```
docker run --runtime=nvidia -e CUDA_VISIBLE_DEVICES=5 --name agents_system -it --init agents_system bash

docker start agents_system

```

Next in the container:

- ``` go to the working directory:
```
cd /projects/llm_agents_chemistry
```
- running Ollama locally
```
ollama serve
```
- download and run the Llama3.1-8b model locally for further running through the Ollama service. The following command should download and run the model
```
ollama run llama3.1
```

## Run

The following commands (from the /projects/llm_agents_chemistry directory) should run the model:
```
conda activate llm_pipeline
python multi_agents_system/inference.py
```

## Run without web interface

Script to run without a web interface:
````python
from chain import Chain

with open(“multi_agents_system/config.yaml”, “r”) as file:
    config = yaml.safe_load(file)

chain = Chain(
    conductor_model=config[“conductor_model”],
    api_vse_gpt=config[“api_vse_gpt”],
    is_many_funcs=bool(config[“is_many_funcs”]),
    attempt=int(config[“attemps”]),
    url=config[“url”]
)

questions = [
    “HERE IS YOUR QUESTION 1”, ‘QUESTION 2’, ”...”
]

    For q in questions:
        response = chain.run(q)
        print('PROCESS: Task complited')
        print(response)
```

## Description of multi_agents_system structure

The script for the multi-agents system is in the current folder, multi_agents_system:

- 'chain.py' - the whole chain of the system, logic for calling agents, valdation of responses, calling functions offered by the agent
- 'agents.py' - classes of agents, construction of personal prompts, processing of responses
- 'memory.py' - chat memory
- 'tools.py' - functions available for the agent to call
- 'inference.py' - launching the interface via gradio