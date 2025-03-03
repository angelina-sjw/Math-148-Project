# Math 148 Project - Yelp Search Engine

## Overview

This project is a search engine for Yelp reviews. It uses a combination of a vector database, LLMs and ViT to search for reviews based on a given query.

## Getting Started

1. Clone the repository

```bash
git clone https://github.com/math148-project/yelp-search-engine.git
```

2. Install the dependencies

The following installs main application dependencies.
```bash
poetry install
```

The following installs clip model server dependencies. Note: clip model server is required to run on localhost for the system.
```bash
cd clip-server
poetry install
```

3. Run the project

Under rec-engine/clip-server, run the following command to start the clip model server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Under rec-engine, run the following command to search:
Basic text query:
```bash
python -m rec_engine.app.main --query "your-query-here" --openai_api_key your-openai-api-key-here
```

Basic image query:
1. Place/replace the image you want to use in image_query folder under rec-engine/rec_engine/app/image_query
2. Run the following command:
```bash
python -m rec_engine.app.main --query "rec_engine/app/image_query/{image_file_name}.jpg" --openai_api_key your-openai-api-key-here
```

Advance text query:
This function implements a keyword extraction agent that prefilters vector search results based on the query.
```bash
python -m rec_engine.app.main --query "your-query-here" --openai_api_key your-openai-api-key-here --advanced_mode
```

