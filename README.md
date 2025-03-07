# Math 148 Project – Yelp Review Usefulness Prediction and Restaurant Image Analysis

This repository contains the code and experiments for our project, which leverages Yelp’s multimodal data to predict review usefulness and analyze restaurant images. The project is divided into three main components:

1. **Text Analysis**  
2. **Image Analysis**  
3. **Multimodal Recommendation System**

The primary dataset used in this project, the Yelp Open Dataset, can be obtained from [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/).  

---

## 1. Text Analysis

The **text-analysis** module processes Yelp review data to predict whether a review will be perceived as "more useful" or "less useful." It employs a range of natural language processing techniques and machine learning models to extract actionable insights from textual reviews.

### Overview

- **Data Download & Preparation:** Use the script in `utils/loading.py` to convert raw JSON files into a cleaned CSV format. This script also handles linking reviews with business and user profiles.
  
- **Feature Extraction:**  
  - **Sentiment Analysis:** Compute sentiment polarity using TextBlob.
  - **Topic Modeling:** Use BERTopic to extract semantic topics from reviews.
  - **TF-IDF Vectorization:** Generate weighted features capturing the importance of unigrams and bigrams.

- **Modeling:**  
  - **Baseline Models:**  
    - *Logistic Regression* on structured numerical data.
  - **Text-Only Models:**  
    - Fine-tuned BERT model.
    - Custom LSTM network.
  - **Integrated Multi-Input Model:**  
    - Combines text embeddings (via an LSTM branch) and numerical features (via dense layers) for enhanced prediction.
    - Uses SMOTE for handling class imbalance.
    - Optimized using binary cross-entropy loss and the Adam optimizer.

### Setup & Execution

#### Prerequisites

- Python 3.8+
- HuggingFace API Token (for access to BERT and BERTopic)
- Required Python packages (see `requirements.txt`)

#### Installation & Running

1. **Download the Yelp Dataset:**  
   Visit [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) and download the necessary JSON files.

2. **Prepare the Data:**  
   Navigate to the `text-analysis` folder and run:
   ```bash
   python utils/loading.py
   ```

This will process the raw JSON data into a CSV file ready for analysis.

3. **Run the Main Notebook:**  
   Open the main notebook (located in the `text-analysis` folder) to:
   - Import helper functions from the `utils` folder.
   - Train and evaluate various models (baseline, BERT, LSTM, and the multi-input neural network).
   - Visualize metrics such as ROC-AUC, Precision-Recall, and model interpretability using LIME.

---

## 2. Image Analysis

The **food-classification** module contains code for processing and analyzing restaurant images. This part of the project focuses on two main tasks: image classification and price prediction using restaurant photos.

### Overview

- **Label Classification:**  
  - Classify images into predefined categories (food, drink, interior, exterior, menu) using a modified ResNet-18.
  
- **Food Classification:**  
  - Fine-tune an EfficientNet-B0 model on the Food101 dataset to assign pseudo-labels to Yelp food images.
  
- **Price Prediction:**  
  - Predict pricing tiers (“Cheap” vs. “Expensive”) based on image features.
  - Compare single-modal (image-only) and multimodal approaches (combining image features with text summaries).

- **Model Explainability**
  - Visualize high to low confidence photo classification.
  - Apply Grad-CAM to highlight important image regions influencing classification decisions.

### Setup & Execution

#### Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- GPU for training
  
#### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/angelina-sjw/Math-148-Project.git
   cd Math-148-Project
   cd food-classification
   ```

2. **Install Dependencies and Activate Virtual Environment:**
   ```bash
   poetry install
   poetry shell
   ```
   
3. **Prepare the Yelp Dataset:**  
   Visit [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) and download the necessary JSON files.
   Place the downloaded dataset under a new `data` directory inside the `food-classification` folder.
   Resize the yelp photos into 224x224 using the `resize_images` function in `data_utils.utils`.

4. **Model Finetuning**
   The `notebooks` directory contains three subdirectories: `label`, `price`, and `food101`, each corresponding to a different classification task. Each subdirectory includes training notebooks prefixed with `train_`, which are used to finetune the models.

5. **Model Explainability**  
   The `explainability` subdirectory contains notebooks for applying Grad-CAM, visualizing predictions, and performing exploratory image data analysis.

6. **Additional Code & Utilities**  
   - All trained models, including the fusion model, ResNet fine-tuning, and Grad-CAM, are stored in the `model` directory.  
   - The PyTorch dataset class and helper functions, such as `stratified_split_dataset`, are located in `data_utils/dataset`.  
   - Data-related utility functions are in `data_utils/utils`, while modeling-related helper functions are in `model/utils`.   

---

## 3. Multimodal Recommendation System

The **rec-engine** module implements a multimodal search engine for Yelp restaurants by integrating textual and visual data.

### Overview

The recommendation engine combines:
- **Vector Database:** Elasticsearch stores and retrieves restaurant embeddings.
- **Large Language Models:** OpenAI’s GPT (or similar) models are used for keyword extraction and understanding natural language queries.
- **CLIP-ViT:** Bridges text and image modalities by generating shared embeddings for both.

This system allows users to search for restaurants using:
- **Text Queries:** Find restaurants based on natural language descriptions.
- **Image Queries:** Upload an image to retrieve visually similar restaurants.
- **Advanced Filtering:** Use AI-extracted keywords to filter results by cuisine, price range, and rating.

### Getting Started

#### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Elasticsearch (local or remote instance; Docker is recommended)
- OpenAI API key (for LLM functionality)
- CUDA-compatible GPU (recommended for running the CLIP model server)
- Docker (for Elasticsearch setup)

#### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/angelina-sjw/Math-148-Project.git
   cd Math-148-Project
   ```

2. **Install Dependencies for Main Application:**
   ```bash
   poetry install
   ```

3. **Setup the CLIP Model Server:**
   ```bash
   cd rec-engine/clip-server
   poetry install
   ```

4. **Add Yelp Dataset Files:**
   In the `rec-engine` folder, create a `data` directory and add the following:
   - `photos/` (folder containing business photos)
   - `photos.json`
   - `yelp_academic_dataset_business.json`

5. **Launch Elasticsearch with Docker:**
   ```bash
   cd rec-engine/docker
   docker-compose up -d
   ```
6. **Load Data into Elasticsearch:**
   ```bash
   python -m rec-engine.app.load_data --openai_api_key YOUR_OPENAI_API_KEY
   ```
### Usage

#### Starting the CLIP Server

Start the embedding server:
   ```bash
   cd rec-engine/clip-server
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Text Search

Make sure both the Elasticsearch container and the CLIP server are running, then execute:
   ```bash
   python -m rec-engine.app.main --query "romantic Italian restaurant for dinner" --openai_api_key YOUR_OPENAI_API_KEY
   ```

For advanced text search with keyword extraction:
   ```bash
   python -m rec-engine.app.main --query "romantic Italian restaurant for dinner" --openai_api_key YOUR_OPENAI_API_KEY --advanced_mode ```
   ```

#### Image Search

1. Place Your Query Image:  
   Put your image in rec-engine/rec_engine/app/image_query/ (e.g., your-food-image.jpg).

2. Run the Image Search:
   ```bash
   python -m rec-engine.app.main --query "rec-engine/rec_engine/app/image_query/your-food-image.jpg" --openai_api_key YOUR_OPENAI_API_KEY
   ```
*Tip:* Use quotes around file paths that contain spaces.

#### Fine-Tuning the CLIP Model

To fine-tune the CLIP model on the Yelp dataset:
```bash
cd rec-engine/clip-server
python train.py --data_path PATH_TO_YOUR_TRAINING_DATA
```

#### Evaluation

Run evaluation scripts to assess system performance:
```bash
python -m tests.evaluation --openai_api_key YOUR_OPENAI_API_KEY
```

---

## Contributing

Contributions, suggestions, and bug reports are welcome. Please open an issue or submit a pull request.

---

## Acknowledgements

- Special thanks to our professor, Lara Kassab, and TAs Joyce Chew and Chi-Hao Wu for their invaluable guidance.
- We appreciate all the team members who contributed to the success of this project.

---

For more details, please feel free to explore the code in each folder.

