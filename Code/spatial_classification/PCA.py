import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SpatialTextDataset(Dataset):
    """Custom dataset for spatial text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTSpatialClassifier(nn.Module):
    """BERT-based classifier with frozen encoder and trainable head"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout_rate=0.3):
        super(BERTSpatialClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        # Get BERT embeddings
        with torch.no_grad():  # Ensure BERT remains frozen
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        if return_embeddings:
            return pooled_output
        
        # Apply dropout and classification head
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits

class SpatialTextClassifier:
    """Main classifier class with PCA analysis capabilities"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BERTSpatialClassifier(model_name, num_classes).to(self.device)
        self.num_classes = num_classes
        self.pca = None
        self.scaler = None
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
        
    def prepare_data(self, csv_path, text_col='text', label_col='label', test_size=0.2):
        """Load and prepare data"""
        # Load data
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Label distribution:\n{df[label_col].value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_col].values,
            df[label_col].values,
            test_size=test_size,
            random_state=42,
            stratify=df[label_col]
        )
        
        # Create datasets
        train_dataset = SpatialTextDataset(X_train, y_train, self.tokenizer)
        test_dataset = SpatialTextDataset(X_test, y_test, self.tokenizer)
        
        return train_dataset, test_dataset
    
    def train(self, train_dataset, val_dataset=None, epochs=5, batch_size=16, learning_rate=2e-4):
        """Train the classifier"""
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # Setup optimizer and loss function
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=learning_rate,
            weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training')
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                val_accuracy = self.evaluate(val_loader)
                val_accuracies.append(val_accuracy)
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}')
            else:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}')
        
        return train_losses, val_accuracies
    
    def extract_embeddings(self, dataset):
        """Extract BERT embeddings from dataset"""
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Extracting embeddings'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get BERT embeddings (CLS token)
                embeddings = self.model(input_ids, attention_mask, return_embeddings=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        labels = np.hstack(all_labels)
        
        return embeddings, labels
    
    def fit_pca(self, embeddings, n_components=50):
        """Fit PCA on embeddings"""
        print(f"Fitting PCA with {n_components} components...")
        
        # Standardize embeddings
        self.scaler = StandardScaler()
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Fit PCA
        self.pca = PCA(n_components=n_components)
        embeddings_pca = self.pca.fit_transform(embeddings_scaled)
        
        # Print explained variance
        explained_var = self.pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        print(f"First 10 components explain {cumsum_var[9]:.3f} of variance")
        print(f"Total variance explained by {n_components} components: {cumsum_var[-1]:.3f}")
        
        return embeddings_pca
    
    def transform_pca(self, embeddings):
        """Transform embeddings using fitted PCA"""
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA not fitted. Please call fit_pca first.")
        
        embeddings_scaled = self.scaler.transform(embeddings)
        return self.pca.transform(embeddings_scaled)
    
    def plot_pca_analysis(self, embeddings_pca, labels, title_prefix=""):
        """Create comprehensive PCA visualizations"""
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{title_prefix}PCA Analysis of BERT Embeddings', fontsize=16)
        
        # 1. Explained Variance
        axes[0, 0].plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                       np.cumsum(self.pca.explained_variance_ratio_))
        axes[0, 0].set_xlabel('Number of Components')
        axes[0, 0].set_ylabel('Cumulative Explained Variance')
        axes[0, 0].set_title('Explained Variance Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 2D PCA Scatter Plot
        colors = ['blue', 'red']
        labels_text = ['Non-Spatial', 'Spatial']
        
        for i, (color, label_text) in enumerate(zip(colors, labels_text)):
            mask = labels == i
            axes[0, 1].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                             c=color, label=label_text, alpha=0.6, s=30)
        
        axes[0, 1].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.3f})')
        axes[0, 1].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.3f})')
        axes[0, 1].set_title('2D PCA Visualization')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 3D PCA Plot (projected to 2D)
        from mpl_toolkits.mplot3d import Axes3D
        axes[1, 0].remove()
        ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
        
        for i, (color, label_text) in enumerate(zip(colors, labels_text)):
            mask = labels == i
            ax_3d.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                         embeddings_pca[mask, 2], c=color, label=label_text, alpha=0.6, s=20)
        
        ax_3d.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.3f})')
        ax_3d.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.3f})')
        ax_3d.set_zlabel(f'PC3 ({self.pca.explained_variance_ratio_[2]:.3f})')
        ax_3d.set_title('3D PCA Visualization')
        ax_3d.legend()
        
        # 4. Component weights heatmap (top 10 components)
        component_weights = self.pca.components_[:10, :50]  # First 10 components, first 50 features
        im = axes[1, 1].imshow(component_weights, cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Principal Component')
        axes[1, 1].set_title('PCA Component Weights (Top 10)')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def analyze_embeddings_with_pca(self, dataset, n_components=50, plot=True):
        """Complete PCA analysis of embeddings"""
        print("Extracting embeddings...")
        embeddings, labels = self.extract_embeddings(dataset)
        
        print(f"Original embedding shape: {embeddings.shape}")
        
        # Fit PCA
        embeddings_pca = self.fit_pca(embeddings, n_components)
        print(f"PCA embedding shape: {embeddings_pca.shape}")
        
        if plot:
            self.plot_pca_analysis(embeddings_pca, labels)
        
        return embeddings_pca, labels
    
    def get_pca_embeddings_for_texts(self, texts):
        """Get PCA-transformed embeddings for new texts"""
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA not fitted. Please call analyze_embeddings_with_pca first.")
        
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get BERT embedding
                embedding = self.model(input_ids, attention_mask, return_embeddings=True)
                embeddings.append(embedding.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        # Transform with PCA
        embeddings_pca = self.transform_pca(embeddings)
        
        return embeddings_pca
    
    def cluster_analysis(self, embeddings_pca, labels, n_clusters=None):
        """Perform clustering analysis on PCA embeddings"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        
        if n_clusters is None:
            n_clusters = self.num_classes
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_pca)
        
        # Calculate clustering metrics
        ari = adjusted_rand_score(labels, cluster_labels)
        silhouette = silhouette_score(embeddings_pca, cluster_labels)
        
        print(f"Clustering Analysis (K={n_clusters}):")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print(f"Silhouette Score: {silhouette:.3f}")
        
        # Plot clustering results
        plt.figure(figsize=(15, 5))
        
        # True labels
        plt.subplot(1, 3, 1)
        colors = ['blue', 'red']
        for i, color in enumerate(colors):
            mask = labels == i
            plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                       c=color, label=f'True Class {i}', alpha=0.6, s=30)
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.3f})')
        plt.title('True Labels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cluster labels
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                            c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.3f})')
        plt.title('K-means Clusters')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        
        # Confusion matrix between true and cluster labels
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(labels, cluster_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('True vs Cluster Labels')
        plt.xlabel('Cluster Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.show()
        
        return cluster_labels, ari, silhouette
    
    def pca_feature_importance(self, feature_names=None, top_n=10):
        """Analyze feature importance in PCA components"""
        if self.pca is None:
            raise ValueError("PCA not fitted. Please call analyze_embeddings_with_pca first.")
        
        # Get component loadings
        components = self.pca.components_
        
        plt.figure(figsize=(12, 8))
        
        # Plot top components
        for i in range(min(4, len(components))):
            plt.subplot(2, 2, i+1)
            
            # Get absolute loadings for this component
            loadings = np.abs(components[i])
            top_indices = np.argsort(loadings)[-top_n:]
            
            plt.barh(range(top_n), loadings[top_indices])
            plt.xlabel('Absolute Loading')
            plt.title(f'PC{i+1} - Top {top_n} Features\n'
                     f'Explained Variance: {self.pca.explained_variance_ratio_[i]:.3f}')
            
            if feature_names:
                plt.yticks(range(top_n), [feature_names[idx] for idx in top_indices])
            else:
                plt.yticks(range(top_n), [f'Feature {idx}' for idx in top_indices])
        
        plt.tight_layout()
        plt.show()
        
        return components
    
    def compare_spatial_vs_nonspatial_pca(self, dataset):
        """Compare PCA patterns between spatial and non-spatial texts"""
        embeddings, labels = self.extract_embeddings(dataset)
        
        # Separate by class
        spatial_embeddings = embeddings[labels == 1]
        nonspatial_embeddings = embeddings[labels == 0]
        
        # Fit separate PCAs
        scaler_spatial = StandardScaler()
        scaler_nonspatial = StandardScaler()
        
        spatial_scaled = scaler_spatial.fit_transform(spatial_embeddings)
        nonspatial_scaled = scaler_nonspatial.fit_transform(nonspatial_embeddings)
        
        pca_spatial = PCA(n_components=20)
        pca_nonspatial = PCA(n_components=20)
        
        spatial_pca = pca_spatial.fit_transform(spatial_scaled)
        nonspatial_pca = pca_nonspatial.fit_transform(nonspatial_scaled)
        
        # Plot comparison
        plt.figure(figsize=(15, 5))
        
        # Explained variance comparison
        plt.subplot(1, 3, 1)
        plt.plot(range(1, 21), np.cumsum(pca_spatial.explained_variance_ratio_), 
                'b-', label='Spatial', linewidth=2)
        plt.plot(range(1, 21), np.cumsum(pca_nonspatial.explained_variance_ratio_), 
                'r-', label='Non-Spatial', linewidth=2)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2D scatter plots
        plt.subplot(1, 3, 2)
        plt.scatter(spatial_pca[:, 0], spatial_pca[:, 1], 
                   c='blue', label='Spatial', alpha=0.6, s=30)
        plt.xlabel(f'PC1 ({pca_spatial.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca_spatial.explained_variance_ratio_[1]:.3f})')
        plt.title('Spatial Texts PCA')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.scatter(nonspatial_pca[:, 0], nonspatial_pca[:, 1], 
                   c='red', label='Non-Spatial', alpha=0.6, s=30)
        plt.xlabel(f'PC1 ({pca_nonspatial.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca_nonspatial.explained_variance_ratio_[1]:.3f})')
        plt.title('Non-Spatial Texts PCA')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return pca_spatial, pca_nonspatial, spatial_pca, nonspatial_pca
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy
    
    def predict(self, texts):
        """Make predictions on new texts"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get prediction
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)
                
                predictions.append(pred.cpu().item())
                probabilities.append(probs.cpu().numpy()[0])
        
        return predictions, probabilities
    
    def save_model(self, save_path='spatial_classifier_model'):
        """Save the trained model, tokenizer, and PCA"""
        import os
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': self.model.bert.config.name_or_path,
                'num_classes': self.num_classes,
            }
        }, os.path.join(save_path, 'model.pth'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save PCA and scaler
        if self.pca is not None:
            with open(os.path.join(save_path, 'pca_components.pkl'), 'wb') as f:
                pickle.dump({
                    'pca': self.pca,
                    'scaler': self.scaler
                }, f)
            print(f"PCA components saved")
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path='spatial_classifier_model'):
        """Load a saved model with PCA components"""
        import os
        import pickle
        
        # Load model checkpoint
        checkpoint = torch.load(os.path.join(load_path, 'model.pth'), 
                               map_location=self.device)
        
        # Recreate model with saved config
        config = checkpoint['model_config']
        self.model = BERTSpatialClassifier(
            model_name=config['model_name'], 
            num_classes=config['num_classes']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        # Load PCA components if available
        pca_path = os.path.join(load_path, 'pca_components.pkl')
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca_data = pickle.load(f)
                self.pca = pca_data['pca']
                self.scaler = pca_data['scaler']
            print("PCA components loaded")
        
        print(f"Model loaded from {load_path}")

    def detailed_evaluation(self, test_dataset):
        """Detailed evaluation with metrics and visualizations"""
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, 
                                     target_names=['Non-Spatial', 'Spatial'])
        cm = confusion_matrix(true_labels, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Spatial', 'Spatial'],
                   yticklabels=['Non-Spatial', 'Spatial'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return accuracy, report, cm

# Example usage with PCA analysis
def main():
    # Initialize classifier
    classifier = SpatialTextClassifier(model_name='distilbert-base-uncased', num_classes=2)
    
    # Load your trained model (if you have one saved)
    # classifier.load_model('spatial_classifier_model')
    
    # Prepare data
    train_dataset, test_dataset = classifier.prepare_data('./train_set.csv')
    
    # If model not trained, train it
    print("Starting training...")
    train_losses, val_accuracies = classifier.train(
        train_dataset, 
        val_dataset=test_dataset,
        epochs=20,
        batch_size=16,
        learning_rate=2e-4
    )
    
    # PCA Analysis on test set
    print("\n" + "="*50)
    print("PCA ANALYSIS")
    print("="*50)
    
    # Analyze embeddings with PCA
    embeddings_pca, labels = classifier.analyze_embeddings_with_pca(
        test_dataset, 
        n_components=50, 
        plot=True
    )
    
    # Perform clustering analysis
    print("\n" + "-"*30)
    print("CLUSTERING ANALYSIS")
    print("-"*30)
    cluster_labels, ari, silhouette = classifier.cluster_analysis(embeddings_pca, labels)
    
    # Compare spatial vs non-spatial PCA patterns
    print("\n" + "-"*30)
    print("CLASS-SPECIFIC PCA ANALYSIS")
    print("-"*30)
    pca_spatial, pca_nonspatial, spatial_pca, nonspatial_pca = classifier.compare_spatial_vs_nonspatial_pca(test_dataset)
    
    # Feature importance analysis
    print("\n" + "-"*30)
    print("PCA FEATURE IMPORTANCE")
    print("-"*30)
    components = classifier.pca_feature_importance()
    
    # Evaluate model
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    accuracy, report, cm = classifier.detailed_evaluation(test_dataset)
    
    # Save the trained model with PCA
    print("\nSaving model with PCA components...")
    classifier.save_model('spatial_classifier_model')
    
    # Example predictions with PCA embeddings
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS WITH PCA")
    print("="*50)
    
    sample_texts = [
        "Where is the cat located?",
        "Do you like this color?", 
        "What is the position of the chair?",
        "How old are you?",
        "Move the object to the left",
        "I enjoy reading books",
        "The building is next to the park",
        "What's your favorite movie?"
    ]
    
    # Get regular predictions
    predictions, probabilities = classifier.predict(sample_texts)
    
    # Get PCA embeddings for these texts
    pca_embeddings = classifier.get_pca_embeddings_for_texts(sample_texts)
    
    print("Sample Predictions with PCA Embeddings:")
    print("-" * 80)
    for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
        label = "Spatial" if pred == 1 else "Non-Spatial"
        confidence = prob[pred]
        
        print(f"Text: '{text}'")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print(f"PCA Embedding (first 5 components): {pca_embeddings[i][:5]}")
        print("-" * 80)
    
    # Visualize sample predictions in PCA space
    plt.figure(figsize=(10, 8))
    
    # Plot test data as background
    colors = ['lightblue', 'lightcoral']
    for i, color in enumerate(colors):
        mask = labels == i
        plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                   c=color, alpha=0.3, s=20, label=f'Test Data - Class {i}')
    
    # Plot sample predictions
    pred_colors = ['blue', 'red']
    for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
        color = pred_colors[pred]
        plt.scatter(pca_embeddings[i, 0], pca_embeddings[i, 1], 
                   c=color, s=100, marker='x', linewidth=3, 
                   label=f'Sample {i+1}' if i < 2 else "")
        
        # Add text annotation
        plt.annotate(f'{i+1}', 
                    (pca_embeddings[i, 0], pca_embeddings[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')
    
    plt.xlabel(f'PC1 ({classifier.pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({classifier.pca.explained_variance_ratio_[1]:.3f})')
    plt.title('Sample Predictions in PCA Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print sample text legend
    print("\nSample Text Legend:")
    for i, text in enumerate(sample_texts):
        print(f"{i+1}: {text}")

def analyze_pca_only(model_path='spatial_classifier_model', data_path='./train_set.csv'):
    """Function to only run PCA analysis on pre-trained model"""
    print("Loading pre-trained model for PCA analysis...")
    
    # Initialize and load model
    classifier = SpatialTextClassifier(model_name='distilbert-base-uncased', num_classes=2)
    classifier.load_model(model_path)
    
    # Prepare data
    _, test_dataset = classifier.prepare_data(data_path)
    
    # Run PCA analysis
    print("\n" + "="*50)
    print("PCA ANALYSIS ON PRE-TRAINED MODEL")
    print("="*50)
    
    embeddings_pca, labels = classifier.analyze_embeddings_with_pca(
        test_dataset, 
        n_components=50, 
        plot=True
    )
    
    # Clustering analysis
    cluster_labels, ari, silhouette = classifier.cluster_analysis(embeddings_pca, labels)
    
    # Class-specific analysis
    classifier.compare_spatial_vs_nonspatial_pca(test_dataset)
    
    # Feature importance
    classifier.pca_feature_importance()
    
    return classifier

if __name__ == "__main__":
    # For full training + PCA
    main()
    
    # Or for PCA analysis only on pre-trained model:
    # classifier = analyze_pca_only()