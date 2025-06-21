# Tourist Attraction Recommendation System

A hybrid recommendation system for tourist attractions in Nepal that combines collaborative filtering, content-based filtering, and deep learning approaches to provide personalized recommendations.

## Features

- Neural Collaborative Filtering (NCF) for user-item interactions
- Content-based recommendations using DistilBERT embeddings
- Hybrid recommendation system combining multiple approaches
- Frontend interface for user interaction
- Dynamic model updates based on new user interactions

## Project Structure

```
├── Data/                      # Processed and raw data files
├── Model/                     # Model implementations
│   ├── ContentBased/         # Content-based recommendation model
│   ├── DistilBert/          # DistilBERT-based recommendation model
│   ├── NCF/                 # Neural Collaborative Filtering implementation
│   └── Hybrid/             # Hybrid recommendation system
├── Frontend/                 # Web interface implementation
├── Notebook/                 # Jupyter notebooks for analysis and experiments
├── utils/                    # Utility functions and helpers
└── requirements.txt          # Python dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the data:
```bash
python Data/Code/clean.py
```

4. Train the models:
```bash
python Model/main.py
```

5. Run the web interface:
```bash
cd Frontend
npm install
npm run dev
```

## Models

### Neural Collaborative Filtering (NCF)
- Implements a hybrid neural network combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
- Trained on user-item interactions
- Supports dynamic updates for online learning

### Content-Based Recommender
- Uses DistilBERT embeddings for attraction descriptions
- Categorized tag-based filtering
- Province and category-based recommendations

### Hybrid Recommender
- Combines predictions from multiple models
- Weighted ensemble approach
- Adaptable weights based on performance

## Data

The system uses multiple data sources:
- User-attraction interaction data (anonymized)
- Attraction metadata and descriptions
- Categorized tags for attractions
- Province and location information

### Dataset Information
This dataset contains tourist attraction information from Nepal and includes:

Main data files:
- `Data.csv`: Tourist attractions with descriptions and locations
- `all_ratings.csv`: Anonymized user ratings
- `users_anonymized.csv`: Anonymized user information
- `CategorizedTags.json`: Structured tag categories

Supporting files:
- `province.txt`: List of provinces
- `UniqueTags.txt`: List of unique tags used in the dataset

### Data Usage and License
The dataset is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). You are free to:
- Share: Copy and redistribute the material in any medium or format
- Adapt: Remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit
- NonCommercial: You may not use the material for commercial purposes
- ShareAlike: If you remix, transform, or build upon the material, you must distribute your contributions under the same license

### Data Privacy
- All user data has been anonymized
- Personal identifiers have been removed
- Location data is limited to public tourist attractions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Tourist attraction data collection team
- Contributors to the model implementations
- Frontend development team
