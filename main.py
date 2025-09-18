# Fashion Outfit Recommendation System + Trend Analyzer Dashboard

## Project Structure
```
fashion-recommendation-system/
│
├── config.yaml
├── requirements.txt
├── README.md
├── dashboard.py
├── data_preprocessing.py
├── model_training.py
├── trend_analysis.py
├── recommendation_engine.py
├── utils.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
└── assets/
    └── styles.css
```

## 1. requirements.txt
```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
tensorflow==2.13.0
torch==2.0.1
transformers==4.30.0
nltk==3.8.1
spacy==3.6.0
opencv-python==4.8.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
wordcloud==1.9.2
Pillow==10.0.0
requests==2.31.0
beautifulsoup4==4.12.2
tweepy==4.14.0
faiss-cpu==1.7.4
pyyaml==6.0
tqdm==4.65.0
sentence-transformers==2.2.2
```

## 2. config.yaml
```yaml
# Configuration file for Fashion Recommendation System

# Data paths
data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  models_path: "data/models/"
  fashion_dataset: "fashion_items.csv"
  user_preferences: "user_preferences.csv"
  trends_data: "fashion_trends.csv"

# Model parameters
model:
  embedding_dim: 128
  learning_rate: 0.001
  batch_size: 32
  epochs: 10
  similarity_threshold: 0.7

# Recommendation parameters
recommendation:
  num_recommendations: 10
  diversity_factor: 0.3
  trend_weight: 0.4
  personalization_weight: 0.6

# Trend analysis
trends:
  keywords_to_track:
    - "#OOTD"
    - "#streetwear"
    - "#fashion"
    - "#style"
    - "#outfitoftheday"
    - "#fashionista"
    - "#vintage"
    - "#sustainable"
  sentiment_threshold: 0.5
  trend_window_days: 7

# API Keys (placeholder - replace with actual keys)
api:
  twitter_bearer_token: "YOUR_TWITTER_BEARER_TOKEN"
  instagram_token: "YOUR_INSTAGRAM_TOKEN"
```

## 3. utils.py
```python
"""
Utility functions for the Fashion Recommendation System
"""

import os
import json
import yaml
import pickle
import logging
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def create_directories():
    """Create necessary directory structure"""
    directories = [
        "data", "data/raw", "data/processed", "data/models",
        "assets", "logs", "cache"
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure created successfully")

def download_dataset(url: str, destination: str) -> bool:
    """Download dataset from URL"""
    try:
        logger.info(f"Downloading dataset from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        logger.info(f"Dataset downloaded to {destination}")
        return True
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False

def save_model(model, filepath: str):
    """Save trained model to disk"""
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def load_model(filepath: str):
    """Load trained model from disk"""
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def calculate_color_similarity(color1: str, color2: str) -> float:
    """Calculate similarity between two colors"""
    # Convert hex to RGB if needed
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Simple Euclidean distance in RGB space
    try:
        rgb1 = hex_to_rgb(color1) if isinstance(color1, str) else color1
        rgb2 = hex_to_rgb(color2) if isinstance(color2, str) else color2
        
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
        similarity = 1 - (distance / (255 * np.sqrt(3)))  # Normalize
        return similarity
    except:
        return 0.5  # Default similarity if error

def generate_cache_key(*args) -> str:
    """Generate cache key from arguments"""
    key_str = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_time_features() -> Dict[str, Any]:
    """Get current time-based features"""
    now = datetime.now()
    return {
        'season': get_season(),
        'day_of_week': now.strftime('%A'),
        'time_of_day': get_time_of_day(),
        'month': now.month,
        'is_weekend': now.weekday() >= 5
    }

def get_season() -> str:
    """Get current season"""
    month = datetime.now().month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def get_time_of_day() -> str:
    """Get time of day category"""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

# Create directories when module is imported
create_directories()
```

## 4. data_preprocessing.py
```python
"""
Data preprocessing module for Fashion Recommendation System
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import logging
from utils import load_config, logger, create_directories

class FashionDataPreprocessor:
    """Preprocessor for fashion datasets"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        create_directories()
        
    def download_fashion_datasets(self):
        """Download and prepare fashion datasets"""
        logger.info("Downloading fashion datasets...")
        
        # Create synthetic fashion dataset if not available
        self.create_synthetic_fashion_data()
        
        # Download Fashion MNIST (lightweight alternative)
        self.download_fashion_mnist()
        
    def create_synthetic_fashion_data(self):
        """Create synthetic fashion dataset for demonstration"""
        logger.info("Creating synthetic fashion dataset...")
        
        # Categories and attributes
        categories = ['shirt', 'dress', 'pants', 'jacket', 'shoes', 'bag', 'accessories']
        colors = ['black', 'white', 'blue', 'red', 'green', 'gray', 'beige', 'pink', 'yellow', 'brown']
        styles = ['casual', 'formal', 'sporty', 'vintage', 'streetwear', 'elegant', 'boho', 'minimalist']
        occasions = ['work', 'party', 'casual', 'wedding', 'date', 'sports', 'beach', 'travel']
        materials = ['cotton', 'silk', 'denim', 'leather', 'wool', 'polyester', 'linen']
        patterns = ['solid', 'striped', 'floral', 'checkered', 'polka', 'abstract', 'geometric']
        brands = ['Zara', 'H&M', 'Nike', 'Adidas', 'Gucci', 'Prada', 'Uniqlo', 'Forever21']
        
        # Generate synthetic data
        num_items = 5000
        fashion_data = []
        
        for i in range(num_items):
            item = {
                'item_id': f'ITEM_{i:05d}',
                'category': np.random.choice(categories),
                'color': np.random.choice(colors),
                'secondary_color': np.random.choice(colors),
                'style': np.random.choice(styles),
                'occasion': np.random.choice(occasions),
                'material': np.random.choice(materials),
                'pattern': np.random.choice(patterns),
                'brand': np.random.choice(brands),
                'price': np.random.uniform(20, 500),
                'rating': np.random.uniform(3.0, 5.0),
                'popularity_score': np.random.uniform(0, 100),
                'sustainability_score': np.random.uniform(0, 10),
                'trending_score': np.random.uniform(0, 100),
                'season': np.random.choice(['spring', 'summer', 'fall', 'winter', 'all']),
                'gender': np.random.choice(['male', 'female', 'unisex']),
                'size_range': np.random.choice(['XS-M', 'M-L', 'L-XL', 'XS-XL']),
                'description': f"Stylish {np.random.choice(styles)} {np.random.choice(categories)} perfect for {np.random.choice(occasions)}"
            }
            fashion_data.append(item)
        
        # Save to CSV
        df = pd.DataFrame(fashion_data)
        df.to_csv(os.path.join(self.config['data']['raw_path'], 'fashion_items.csv'), index=False)
        logger.info(f"Created synthetic fashion dataset with {num_items} items")
        
        # Create user preferences dataset
        self.create_user_preferences_data()
        
        # Create fashion trends dataset
        self.create_fashion_trends_data()
        
    def create_user_preferences_data(self):
        """Create synthetic user preferences dataset"""
        logger.info("Creating user preferences dataset...")
        
        num_users = 1000
        num_interactions = 10000
        
        user_data = []
        for i in range(num_interactions):
            user_data.append({
                'user_id': f'USER_{np.random.randint(0, num_users):04d}',
                'item_id': f'ITEM_{np.random.randint(0, 5000):05d}',
                'interaction_type': np.random.choice(['view', 'like', 'purchase', 'save']),
                'rating': np.random.uniform(1, 5),
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
            })
        
        df = pd.DataFrame(user_data)
        df.to_csv(os.path.join(self.config['data']['raw_path'], 'user_preferences.csv'), index=False)
        logger.info(f"Created user preferences dataset with {num_interactions} interactions")
        
    def create_fashion_trends_data(self):
        """Create synthetic fashion trends dataset"""
        logger.info("Creating fashion trends dataset...")
        
        trends_keywords = [
            'oversized blazers', 'cottage core', 'dark academia', 'y2k fashion',
            'sustainable fashion', 'minimalist wardrobe', 'street style', 'vintage denim',
            'athleisure', 'monochrome outfits', 'bold prints', 'pastel colors',
            'leather jackets', 'midi skirts', 'chunky sneakers', 'bucket hats'
        ]
        
        trends_data = []
        for _ in range(500):
            trend = {
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30)),
                'keyword': np.random.choice(trends_keywords),
                'platform': np.random.choice(['twitter', 'instagram', 'pinterest', 'tiktok']),
                'mentions': np.random.randint(10, 1000),
                'sentiment': np.random.uniform(-1, 1),
                'engagement_rate': np.random.uniform(0, 10),
                'region': np.random.choice(['US', 'UK', 'EU', 'Asia', 'Global'])
            }
            trends_data.append(trend)
        
        df = pd.DataFrame(trends_data)
        df.to_csv(os.path.join(self.config['data']['raw_path'], 'fashion_trends.csv'), index=False)
        logger.info("Created fashion trends dataset")
        
    def download_fashion_mnist(self):
        """Download Fashion MNIST dataset"""
        try:
            logger.info("Loading Fashion MNIST dataset...")
            fashion_mnist = tf.keras.datasets.fashion_mnist
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
            
            # Normalize images
            train_images = train_images / 255.0
            test_images = test_images / 255.0
            
            # Save processed data
            np.save(os.path.join(self.config['data']['processed_path'], 'fashion_mnist_train_images.npy'), train_images)
            np.save(os.path.join(self.config['data']['processed_path'], 'fashion_mnist_train_labels.npy'), train_labels)
            np.save(os.path.join(self.config['data']['processed_path'], 'fashion_mnist_test_images.npy'), test_images)
            np.save(os.path.join(self.config['data']['processed_path'], 'fashion_mnist_test_labels.npy'), test_labels)
            
            logger.info("Fashion MNIST dataset downloaded and processed")
        except Exception as e:
            logger.error(f"Error downloading Fashion MNIST: {e}")
            
    def preprocess_fashion_items(self) -> pd.DataFrame:
        """Preprocess fashion items dataset"""
        logger.info("Preprocessing fashion items...")
        
        # Load data
        df = pd.read_csv(os.path.join(self.config['data']['raw_path'], 'fashion_items.csv'))
        
        # Encode categorical variables
        categorical_cols = ['category', 'color', 'style', 'occasion', 'material', 'pattern', 'brand', 'season', 'gender']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Normalize numerical features
        numerical_cols = ['price', 'rating', 'popularity_score', 'sustainability_score', 'trending_score']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        # Save processed data
        df.to_csv(os.path.join(self.config['data']['processed_path'], 'fashion_items_processed.csv'), index=False)
        logger.info("Fashion items preprocessed and saved")
        
        return df
    
    def preprocess_user_preferences(self) -> pd.DataFrame:
        """Preprocess user preferences dataset"""
        logger.info("Preprocessing user preferences...")
        
        # Load data
        df = pd.read_csv(os.path.join(self.config['data']['raw_path'], 'user_preferences.csv'))
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create user-item interaction matrix
        interaction_matrix = df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        # Save processed data
        interaction_matrix.to_csv(
            os.path.join(self.config['data']['processed_path'], 'user_item_matrix.csv')
        )
        logger.info("User preferences preprocessed and saved")
        
        return interaction_matrix
    
    def extract_color_features(self, image_path: str) -> Dict:
        """Extract color features from fashion image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'dominant_colors': [], 'color_histogram': []}
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for faster processing
            img_small = cv2.resize(img_rgb, (150, 150))
            
            # K-means clustering for dominant colors
            pixels = img_small.reshape(-1, 3)
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
            
            # Color histogram
            hist_r = cv2.calcHist([img_rgb], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([img_rgb], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([img_rgb], [2], None, [32], [0, 256])
            
            color_histogram = np.concatenate([hist_r, hist_g, hist_b]).flatten().tolist()
            
            return {
                'dominant_colors': dominant_colors,
                'color_histogram': color_histogram
            }
        except Exception as e:
            logger.error(f"Error extracting color features: {e}")
            return {'dominant_colors': [], 'color_histogram': []}
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        logger.info("Preparing training data...")
        
        # Load processed fashion items
        df = pd.read_csv(
            os.path.join(self.config['data']['processed_path'], 'fashion_items_processed.csv')
        )
        
        # Select features for training
        feature_cols = [col for col in df.columns if '_encoded' in col or col in 
                       ['price', 'rating', 'popularity_score', 'sustainability_score', 'trending_score']]
        
        X = df[feature_cols].values
        y = df['category_encoded'].values  # Use category as target for demonstration
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        return X_train, X_test, y_train, y_test

# Main execution
if __name__ == "__main__":
    preprocessor = FashionDataPreprocessor()
    preprocessor.download_fashion_datasets()
    preprocessor.preprocess_fashion_items()
    preprocessor.preprocess_user_preferences()
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data()
    logger.info("Data preprocessing complete!")
```

## 5. model_training.py
```python
"""
Model training module for Fashion Recommendation System
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import faiss
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from utils import load_config, save_model, logger
from data_preprocessing import FashionDataPreprocessor

class FashionRecommendationModel:
    """Main model class for fashion recommendations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.embedding_model = None
        self.classification_model = None
        self.similarity_index = None
        self.preprocessor = FashionDataPreprocessor(config_path)
        
    def build_embedding_model(self, input_dim: int) -> Model:
        """Build neural network for fashion item embeddings"""
        logger.info("Building embedding model...")
        
        # Input layer
        inputs = keras.Input(shape=(input_dim,))
        
        # Encoder
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Embedding layer
        embeddings = layers.Dense(self.config['model']['embedding_dim'], name='embeddings')(x)
        
        # Decoder (for reconstruction)
        x = layers.Dense(128, activation='relu')(embeddings)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(input_dim, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=[embeddings, outputs])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['model']['learning_rate']),
            loss=['mse', 'mse'],
            loss_weights=[1.0, 0.5],
            metrics=['mae']
        )
        
        self.embedding_model = model
        logger.info(f"Embedding model built with {model.count_params()} parameters")
        
        return model
    
    def build_classification_model(self) -> RandomForestClassifier:
        """Build classification model for style prediction"""
        logger.info("Building classification model...")
        
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        return self.classification_model
    
    def train_models(self):
        """Train all models"""
        logger.info("Starting model training...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_training_data()
        
        # Train embedding model
        self.train_embedding_model(X_train, X_test)
        
        # Train classification model
        self.train_classification_model(X_train, y_train, X_test, y_test)
        
        # Build similarity index
        self.build_similarity_index(X_train)
        
        logger.info("Model training completed!")
    
    def train_embedding_model(self, X_train: np.ndarray, X_test: np.ndarray):
        """Train the embedding model"""
        if self.embedding_model is None:
            self.build_embedding_model(X_train.shape[1])
        
        logger.info("Training embedding model...")
        
        # Create dummy targets for autoencoder
        y_train_dummy = X_train.copy()
        y_test_dummy = X_test.copy()
        
        # Training with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
        
        history = self.embedding_model.fit(
            X_train,
            [y_train_dummy, X_train],  # Two outputs: embeddings and reconstruction
            validation_data=(X_test, [y_test_dummy, X_test]),
            epochs=self.config['model']['epochs'],
            batch_size=self.config['model']['batch_size'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(self.config['data']['models_path'], 'embedding_model.h5')
        self.embedding_model.save(model_path)
        logger.info(f"Embedding model saved to {model_path}")
        
        return history
    
    def train_classification_model(self, X_train, y_train, X_test, y_test):
        """Train the classification model"""
        if self.classification_model is None:
            self.build_classification_model()
        
        logger.info("Training classification model...")
        
        # Train model
        self.classification_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classification_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        logger.info(f"Classification Model Performance:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1:.3f}")
        
        # Save model
        model_path = os.path.join(self.config['data']['models_path'], 'classification_model.pkl')
        save_model(self.classification_model, model_path)
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        with open(os.path.join(self.config['data']['models_path'], 'metrics.json'), 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
    
    def build_similarity_index(self, X_train: np.ndarray):
        """Build FAISS index for similarity search"""
        logger.info("Building similarity index...")
        
        # Get embeddings from the model
        if self.embedding_model is not None:
            # Create embedding extractor
            embedding_extractor = Model(
                inputs=self.embedding_model.input,
                outputs=self.embedding_model.get_layer('embeddings').output
            )
            embeddings = embedding_extractor.predict(X_train)
        else:
            # Use PCA if embedding model not available
            pca = PCA(n_components=self.config['model']['embedding_dim'])
            embeddings = pca.fit_transform(X_train)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        self.similarity_index = index
        
        # Save index
        index_path = os.path.join(self.config['data']['models_path'], 'similarity_index.faiss')
        faiss.write_index(index, index_path)
        logger.info(f"Similarity index saved with {index.ntotal} vectors")
        
        return index
    
    def get_embeddings(self, features: np.ndarray) -> np.ndarray:
        """Get embeddings for given features"""
        if self.embedding_model is not None:
            embedding_extractor = Model(
                inputs=self.embedding_model.input,
                outputs=self.embedding_model.get_layer('embeddings').output
            )
            return embedding_extractor.predict(features)
        else:
            # Fallback to PCA
            pca = PCA(n_components=self.config['model']['embedding_dim'])
            return pca.fit_transform(features)

class CollaborativeFilteringModel:
    """Collaborative filtering for user-based recommendations"""
    
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        
    def train(self, user_item_matrix: pd.DataFrame, epochs: int = 10):
        """Train collaborative filtering model using matrix factorization"""
        logger.info("Training collaborative filtering model...")
        
        # Convert to numpy array
        R = user_item_matrix.values
        m, n = R.shape
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (m, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n, self.n_factors))
        
        # Simple SGD for matrix factorization
        learning_rate = 0.01
        regularization = 0.01
        
        for epoch in range(epochs):
            for i in range(m):
                for j in range(n):
                    if R[i, j] > 0:  # Only consider non-zero ratings
                        # Compute error
                        prediction = np.dot(self.user_factors[i], self.item_factors[j])
                        error = R[i, j] - prediction
                        
                        # Update factors
