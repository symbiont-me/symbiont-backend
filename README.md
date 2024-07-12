# Symbiont - An Open Source Self-hosted RAG App 🌐

Welcome to Symbiont, the premier RAG (Retrieval Argument Generation)-based application designed for secure, self-hosted deployment on personal or organizational servers. Built with privacy and user control in mind, Symbiont ensures that sensitive data is managed securely without requiring transmission to third-party servers, except when interacting with LLM (Large Language Model) providers such as OpenAI or Anthropic.

🔗 **Explore the Frontend**: For more about the frontend component of the Symbiont app, please visit: [Symbiont Frontend Repo](#)

## Features 🌟

### 🛡️ Enhanced Privacy and Security
Your data remains under your control, securely stored on your own infrastructure.

### 🔑 Multi-user Authentication
Enables secure, personalized access for teams and organizations.

### 📄 Comprehensive Content Management
- **PDF Viewer**: Directly interact with PDFs.
- **Video Viewer**: Stream and analyze video content efficiently.
- **Multimedia Uploads**: Support for various formats including YouTube videos, web pages, and plain text.

### 📝 Integrated Writing and Note-Taking Tool
Facilitates seamless note-taking and document drafting alongside AI interactions.

### 🤖 Support for Multiple LLMs
Works with various Large Language Models from industry leaders such as Anthropic, OpenAI, and Google. More integrations planned.

## Branches 🌿

- **`main`**: Stable branch, uses hosted services like Pinecone for vectors and Firebase for database and storage.
- **`dev`**: Includes the latest features, fully functional with a self-hosted MongoDB for enhanced privacy, still uses Pinecone for vectors.
- **`vector-db`**: Focuses on integrating various vector databases, both self-hosted and cloud-based, for ultimate privacy and control.


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


## Fair Use and Licensing 📜

Symbiont is committed to providing powerful, free software tools that empower individuals, NGOs, and non-commercial entities to utilize advanced technology ethically and effectively. Our use of the Affero GPL license ensures that all derivatives of our work are also kept open and free, fostering a community of sharing and improvement.

### Commercial Use
While we encourage widespread use of Symbiont, commercial entities are expected to contribute back to the community either by participating in development or through a licensing fee. These contributions help maintain Symbiont's sustainability and ensure it remains free for non-commercial users. For more details on commercial licensing, please contact [contact info].

## Use Cases 🛠️

- **Academic Research**: Secure analysis of sensitive data.
- **Journalism**: Confidential information handling for reporting.
- **Creative Writing**: Private brainstorming and draft creation.

## Contributions 🤝

We welcome contributions from all, from code enhancements to documentation updates. Interested in contributing? Please review our [Contribution Guidelines](#).

Join us in our mission to democratize AI applications while maintaining privacy and security. 🌍🚀👩‍💻👨‍💻

