from huggingface_hub import login
login(token="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")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

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
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        with torch.no_grad():  # Ensure BERT remains frozen
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Apply dropout and classification head
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits

class SpatialTextInference:
    """Lightweight inference class for spatial text classification"""
    
    def __init__(self, model_path='spatial_classifier_model'):
        """
        Initialize the inference model
        
        Args:
            model_path (str): Path to the saved model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.max_length = 128
        
        # Load model and tokenizer
        self._load_model()
        
        # print(f"✅ Spatial Text Classifier loaded successfully!")
        # print(f"📱 Device: {self.device}")
        # print(f"🎯 Ready for inference")
        
    def _load_model(self):
        """Load the saved model and tokenizer"""
        import os
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(
                os.path.join(self.model_path, 'model.pth'), 
                map_location=self.device
            )
            
            # Get model config
            config = checkpoint['model_config']
            
            # Initialize model
            self.model = BERTSpatialClassifier(
                model_name=config['model_name'], 
                num_classes=config['num_classes']
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # print(f"✅ Model loaded from: {self.model_path}")
            
        except Exception as e:
            raise Exception(f"❌ Failed to load model: {str(e)}")
    
    def predict_single(self, text: str) -> Tuple[str, float, dict]:
        """
        Predict a single text input
        
        Args:
            text (str): Input text to classify
            
        Returns:
            tuple: (predicted_label, confidence, probabilities_dict)
        """
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert to readable format
        label = "Spatial" if predicted_class == 1 else "Non-Spatial"
        prob_dict = {
            "Non-Spatial": probabilities[0][0].item(),
            "Spatial": probabilities[0][1].item()
        }
        
        return label, confidence, prob_dict
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """
        Predict multiple texts in batches
        
        Args:
            texts (List[str]): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            List[dict]: List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(logits, dim=1)
            
            # Process results
            for j, text in enumerate(batch_texts):
                pred_class = predicted_classes[j].item()
                confidence = probabilities[j][pred_class].item()
                
                label = "Spatial" if pred_class == 1 else "Non-Spatial"
                prob_dict = {
                    "Non-Spatial": probabilities[j][0].item(),
                    "Spatial": probabilities[j][1].item()
                }
                
                batch_results.append({
                    "text": text,
                    "predicted_label": label,
                    "confidence": confidence,
                    "probabilities": prob_dict
                })
            
            results.extend(batch_results)
        
        return results
    
    def predict_csv(self, csv_path: str, text_column: str = 'text', 
                   output_path: str = None) -> pd.DataFrame:
        """
        Predict on a CSV file
        
        Args:
            csv_path (str): Path to input CSV file
            text_column (str): Name of the text column
            output_path (str): Optional path to save results
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        texts = df[text_column].astype(str).tolist()
        
        print(f"🔄 Processing {len(texts)} texts from CSV...")
        
        # Get predictions
        results = self.predict_batch(texts)
        
        # Add predictions to dataframe
        df['predicted_label'] = [r['predicted_label'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        df['prob_non_spatial'] = [r['probabilities']['Non-Spatial'] for r in results]
        df['prob_spatial'] = [r['probabilities']['Spatial'] for r in results]
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"💾 Results saved to: {output_path}")
        
        return df
    
    def interactive_demo(self):
        """Interactive demonstration mode"""
        print("\n🎯 Interactive Spatial Text Classification Demo")
        print("💡 Enter texts to classify (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("⚠️  Please enter some text")
                continue
            
            try:
                label, confidence, probabilities = self.predict_single(text)
                
                print(f"\n📊 Results:")
                print(f"   🏷️  Prediction: {label}")
                print(f"   📈 Confidence: {confidence:.3f}")
                print(f"   📊 Probabilities:")
                print(f"      Non-Spatial: {probabilities['Non-Spatial']:.3f}")
                print(f"      Spatial: {probabilities['Spatial']:.3f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Example usage of the inference class"""
    
    # Initialize inference model
    try:
        classifier = SpatialTextInference(model_path='spatial_classifier_model')
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return
    
    # Example 1: Single prediction
    print("\n🔍 Single Prediction Example:")
    sample_text = "Where is the cat sitting?"
    label, confidence, probs = classifier.predict_single(sample_text)
    
    print(f"Text: '{sample_text}'")
    print(f"Prediction: {label} (confidence: {confidence:.3f})")
    print(f"Probabilities: {probs}")
    
    # Example 2: Batch prediction
    print("\n📦 Batch Prediction Example:")
    sample_texts = [
        "Where do you see the table?",
        "What color do you prefer?",
        "Is the lamp above the desk?",
        "How are you feeling today?",
        "What's the position of the chair?"
    ]
    
    results = classifier.predict_batch(sample_texts)
    
    for result in results:
        print(f"Text: '{result['text']}'")
        print(f"Prediction: {result['predicted_label']} ({result['confidence']:.3f})")
        print("-" * 40)
    
    # Example 3: Interactive mode (uncomment to try)
    # classifier.interactive_demo()

if __name__ == "__main__":
    main()