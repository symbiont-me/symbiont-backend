# Symbiont - An Open Source Self-hosted RAG app

Symbiont is a RAG-based app designed to be self-hosted and run from your own server or computer. The purpose of Symbiont is to allow anyone to be able to run a RAG-based app with a few single commands. 
The entire app can be hosted locally, which means all the data is secure and does not go to a third-party server, except the data you send as part of the prompt to the third-party LLM providers such as OpenAI or Anthropic. 

NOTE: Not all self-hosted features are implemented in the v1 of the app. V1 is still using Pinecone for vector database and firebase for data storage and database. This will be remedied soon with v2 in which both VectorDB and Database will be self-hosted instance, which could be run as Docker containers. 


This repo is for the Backend of the Symbiont app. For the frontend please see: 


Symbiont is still a work in progress.

## Symbiont Backend

### Setup
#### Development Environment
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

___

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
FIREBASE_CREDENTIALS=BASE64 string
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
