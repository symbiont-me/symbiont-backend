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
