# Symbiont Backend

## Setup

### Development Environment

<details>
  <summary>Prepare .env.dev and build image</summary>
     
#### 1. Prepare `.env` file
For development environment, prepare a `.env` file that has the following fields:

```
OPENAI_API_KEY=value
ANTHROPIC_API_KEY=value
PINECONE_API_KEY=value
PINECONE_INDEX=value
PINECONE_API_ENDPOINT=value
PINECONE_REGION=value
FIREBASE_CREDENTIALS=value
GOOGLE_GEMINI_API_KEY=value
VOYAGE_API_KEY=value
CO_API_KEY=value
TOGETHER_API_KEY=value
```

#### 2. Build and Run

Build and run the docker image for dev environment:

```bash
docker-compose --profile dev up
```

This will run the API server at port `0.0.0.0:8000`

</details>

---

### Production Environment

<details>
  <summary>Prepare .env.prod and build image</summary>
  
#### 1. Prepare `.env` file
For production environment, prepare a `.env` file that has the following fields:
```
OPENAI_API_KEY=value
ANTHROPIC_API_KEY=value
PINECONE_API_KEY=value
PINECONE_INDEX=value
PINECONE_API_ENDPOINT=value
PINECONE_REGION=value
FIREBASE_CREDENTIALS=value
GOOGLE_GEMINI_API_KEY=value
VOYAGE_API_KEY=value
CO_API_KEY=value
TOGETHER_API_KEY=value
```
#### 2. Build and Run
Build and run the docker image for dev environment:
```bash
docker-compose --profile prod up
```
This will run the API server at port `0.0.0.0:80`
</details>

#### 3. Running with Local MongoDB instance

- Make sure docker is running
- Run the following commands:

```bash
docker pull mongo
# if running for the first time this will start local mongo instance with a mounted volume for data persistence
docker run -d -p 27017:27017 -v $(pwd)/database:/data/db --name symbiont mongo:latest
# to run an existing container created by the above command
docker start symbiont
```bash
docker run -d --name qdrant_instance -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

