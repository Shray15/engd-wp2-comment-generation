# Intent prediction utility functions for the comment generator app
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os
import re
import nltk
from nltk.corpus import stopwords

# Add the research paper path to system path
sys.path.append(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper") 

try:
    from data.intent_train_test_preprocess import preprocess
except ImportError:
    # Fallback preprocess function if the import fails
    def preprocess(text):
        """Basic text preprocessing function"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip()

# Labels used during training (order must match model head)
labels = ['Appreciation', 'Criticism', 'Inquiry', 'Statement']

# Global variables to store loaded models
_intent_models = None
_intent_tokenizers = None
_models_loaded = False

def load_intent_models():
    """Load all intent prediction models once"""
    global _intent_models, _intent_tokenizers, _models_loaded
    
    if _models_loaded:
        return _intent_models, _intent_tokenizers
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {}
    tokenizers = {}
    
    model_names = ["GRONLP_new3_CV", "robert_new3_CV", "debertaV3_new3_CV"]
    
    for model_name in model_names:
        try:
            model_output_dir = f"C:\\Users\\20245179\\OneDrive - TU Eindhoven\\LLM_EngD_project\\Intent recognition Comments\\Fine_tuned_4_intents\\real_data_distrubution\\models\\fine_tune_BERT\\intent_fine_tuned_4_intents_{model_name}"
            
            if os.path.exists(model_output_dir):
                model = AutoModelForSequenceClassification.from_pretrained(model_output_dir).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_output_dir, use_fast=False)
                model.eval()
                
                models[model_name] = model
                tokenizers[model_name] = tokenizer
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            continue
    
    _intent_models = models
    _intent_tokenizers = tokenizers
    _models_loaded = True
    
    return models, tokenizers

def predict_intent_for_text(text, max_length=512):
    """
    Predict intent for a single text using majority voting from multiple models
    
    Args:
        text (str): The comment text to analyze
        max_length (int): Maximum sequence length for tokenization
    
    Returns:
        dict: Dictionary containing intent prediction results
    """
    global _intent_models, _intent_tokenizers
    
    # Load models if not already loaded
    if not _models_loaded:
        models, tokenizers = load_intent_models()
    else:
        models = _intent_models
        tokenizers = _intent_tokenizers
    
    if not models:
        return {
            "intent": "Unknown", 
            "confidence": 0.0, 
            "error": "No models available"
        }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess the text
    processed_text = preprocess(text)
    
    predictions = {}
    confidences = {}
    
    for model_name, model in models.items():
        try:
            tokenizer = tokenizers[model_name]
            
            # Tokenize
            inputs = tokenizer(
                processed_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits.float()
                probs = torch.softmax(logits, dim=-1)
                
                # Get prediction
                pred_idx = probs.argmax(dim=-1).item()
                pred_label = labels[pred_idx]
                confidence = probs.max().item()
                
                predictions[model_name] = pred_label
                confidences[model_name] = confidence
                
        except Exception as e:
            print(f"Error predicting with model {model_name}: {e}")
            continue
    
    if not predictions:
        return {
            "intent": "Unknown", 
            "confidence": 0.0, 
            "error": "All models failed"
        }
    
    # Majority voting
    from collections import Counter
    vote_counts = Counter(predictions.values())
    
    if vote_counts:
        # Get the most common prediction
        final_intent = vote_counts.most_common(1)[0][0]
        
        # Calculate average confidence for the final intent
        intent_confidences = [conf for model, pred in predictions.items() 
                             if pred == final_intent 
                             for conf in [confidences.get(model, 0.0)]]
        avg_confidence = sum(intent_confidences) / len(intent_confidences) if intent_confidences else 0.0
        
        return {
            "intent": final_intent,
            "confidence": avg_confidence,
            "model_predictions": predictions,
            "model_confidences": confidences,
            "vote_counts": dict(vote_counts)
        }
    else:
        return {
            "intent": "Unknown", 
            "confidence": 0.0, 
            "error": "No valid predictions"
        }

def analyze_intent_batch(texts, batch_size=32, max_length=512):
    """
    Analyze intents for a batch of texts
    
    Args:
        texts (list): List of text strings to analyze
        batch_size (int): Batch size for processing
        max_length (int): Maximum sequence length
    
    Returns:
        list: List of intent prediction dictionaries
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            result = predict_intent_for_text(text, max_length)
            results.append(result)
    
    return results