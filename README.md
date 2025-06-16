# hack-the-sad
little LLM for querying LP info


# done: 
* the bot now calls the ChromaDB
* The script now calls the bot in the container.
* we can process files in a directory

# 2DO:
* pass the result from the tool-call to the LLM to generate an answer.
* pass the bot's answer to the frontend. 

# setup
* start a new venv: python -m venv venv
* run the venv: ./venv/Scripts/Activate.ps1
* install the dependencies: pip install -r requirements.txt
* run the bot! and make requests to the flask endpoint
* 
* process_files.py will chunk files in the given directory and pass them to the chromadb as vectors.
* bot.py will then start a flask app, and redirect the query to the LLM, instantiate some tools, and the rest. 

# useful
* instantiating a new container with Ollama: docker run -d --gpus=all -v $volumneNameInHost:/root/.ollama -p $port:$port --name $name ollama/ollama
* running an llm on Ollama inside a docker container: docker exec -it $name ollama run $model
* information about the model ran by ollama in a container: docker exec -it $name ollama list

# chroma-db setup
* pulling the chroma-db image: docker pull chromadb/chroma:latest
* running the image: docker run -d --name chroma-db -p 8000:8000 -v $(pwd)/chroma_data:/chroma/chroma -e IS_PERSISTENT=TRUE -e PERSIST_DIRECTORY=/chroma/chroma chromadb/chroma:latest
* (OPTIONAL) verifying it is running: curl http://localhost:8000/api/v2/heartbeat
```
import chromadb

# Connect to your ChromaDB instance
client = chromadb.HttpClient(host='localhost', port=8000)

# Test connection
print(client.heartbeat())

# Create a test collection
collection = client.create_collection("test_collection")
print("ChromaDB is working!")
```
