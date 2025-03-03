# Math 148 Project - Yelp Search Engine

## Overview

This project is a multimodal search engine for Yelp restaurants that combines vector embeddings, large language models, and computer vision to provide intelligent search capabilities. The system allows users to search using:

1. **Text queries** - Find restaurants based on natural language descriptions
2. **Image queries** - Find restaurants that visually match an uploaded image
3. **Advanced filtering** - Filter by cuisine, price range, and rating using AI-extracted keywords

## Architecture

The system consists of three main components:

- **Vector Database**: Elasticsearch for storing and searching through restaurant embeddings
- **Large Language Models**: OpenAI's GPT models for keyword extraction and natural language understanding
- **Vision Transformer (ViT)**: CLIP model for image-to-text embeddings and visual similarity search

## Getting Started

### Prerequisites

- Python 3.8+
- Poetry for dependency management
- Elasticsearch (local or remote instance)
- OpenAI API key
- CUDA-compatible GPU (recommended for CLIP model server)
- Docker (required for Elasticsearch)

### Installation

1. Clone the repository

```bash
git clone https://github.com/math148-project/yelp-search-engine.git
cd yelp-search-engine
```

2. Install main application dependencies

```bash
poetry install
```

3. Install CLIP model server dependencies

```bash
cd clip-server
poetry install
```
4. Add Yelp dataset

```bash
cd rec_engine
mkdir data
```
Place following folders/files in the data folder:
- photos/
- photos.json
- yelp_academic_dataset_business.json

Note: photos/ is the folder containing jpg business photos from the origin yelp dataset files.

5. Create Elasticsearch index using docker:

```bash
cd docker
docker-compose up -d
```

### Configuration

1. Configure Elasticsearch (default: `localhost:9200`)
2. Obtain an OpenAI API key for LLM functionality

### Loading Data

Load the Yelp dataset into Elasticsearch:

```bash
python -m rec_engine.app.load_data --openai_api_key your-openai-api-key-here
```

## Usage

### Starting the CLIP Server

This server provides text and image embedding:

```bash
cd rec-engine/clip-server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
### Text Search

Simple text search:

```bash
python -m rec_engine.app.main --query "romantic Italian restaurant for dinner" --openai_api_key your-openai-api-key-here
```

Advanced text search with keyword extraction:

```bash
python -m rec_engine.app.main --query "romantic Italian restaurant for dinner" --openai_api_key your-openai-api-key-here --advanced_mode
```

### Image Search

1. Place your query image in the `image_query` folder:
   ```
   rec-engine/rec_engine/app/image_query/your-food-image.jpg
   ```

2. Run image search:
   ```bash
   python -m rec_engine.app.main --query "rec_engine/app/image_query/your-food-image.jpg" --openai_api_key your-openai-api-key-here
   ```

   **Important**: Use quotes around file paths, especially if they contain spaces.

### Fine-tuning the CLIP Model

Note: Currently only supports fine-tuning on Yelp dataset.
The system can be enhanced with a fine-tuned CLIP model on domain-specific data:

```bash
cd rec-engine/clip-server
python train.py --data_path path/to/your/training/data
```
