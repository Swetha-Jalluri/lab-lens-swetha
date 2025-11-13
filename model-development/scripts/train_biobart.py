"""
BioBART Fine-tuning for Medical Text Summarization
Author: Lab Lens Team
Description: Fine-tunes BioBART on MIMIC-III discharge summaries for abstractive summarization

Model: GanjinZero/biobart-v2-base
Pre-training: PubMed biomedical literature
Fine-tuning: MIMIC-III clinical discharge summaries

This script:
1. Loads prepared train/validation/test datasets
2. Initializes BioBART model and tokenizer
3. Configures training parameters
4. Fine-tunes model with MLflow experiment tracking
5. Evaluates performance using ROUGE metrics on validation and test sets
6. Saves best model for deployment
7. Generates sample summaries for quality review
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# PyTorch and ML libraries
import torch
from torch.utils.data import Dataset

# HuggingFace Transformers for BioBART
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

# Evaluation metrics
from rouge_score import rouge_scorer

# Experiment tracking
import mlflow
import mlflow.pytorch


# Model and Training Configuration
# These parameters control the fine-tuning process
MODEL_NAME = "GanjinZero/biobart-v2-base"  # BioBART pre-trained on PubMed
MAX_INPUT_LENGTH = 512  # Maximum input tokens for discharge summary
MAX_TARGET_LENGTH = 256  # Maximum summary tokens (roughly 1000 characters)
BATCH_SIZE = 2  # Number of samples per batch (smaller for CPU stability)
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients over 2 steps (effective batch size = 4)
LEARNING_RATE = 2e-5  # Learning rate for AdamW optimizer
NUM_EPOCHS = 3  # Number of complete passes through training data
WARMUP_STEPS = 100  # Gradual learning rate warmup
EVAL_STEPS = 500  # Run evaluation every 500 training steps
RANDOM_SEED = 42  # For reproducibility across runs


class MIMICSummarizationDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-III discharge summary summarization.
    Handles tokenization and formatting for BioBART model.
    """
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_input_length: int, max_target_length: int):
        """
        Initialize dataset with tokenizer and length constraints.
        
        Args:
            dataframe: DataFrame with input_text and target_summary columns
            tokenizer: HuggingFace tokenizer (BioBART)
            max_input_length: Maximum input sequence length in tokens
            max_target_length: Maximum target sequence length in tokens
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        print(f"Dataset initialized with {len(self.data)} samples")
        
    def __len__(self):
        """Return total number of samples"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single tokenized training example.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels for model
        """
        row = self.data.iloc[idx]
        
        # Get input and target texts
        input_text = str(row['input_text'])
        target_text = str(row['target_summary'])
        
        # Tokenize input discharge summary
        # Truncation handles variable-length inputs
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target summary
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Return tensors in format expected by model
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }


def load_config(config_path: str = 'model-development/configs/model_config.json') -> Dict:
    """
    Load model configuration from JSON file.
    
    Args:
        config_path: Path to model configuration file
        
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_path}")
        return config
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        return {}


def load_datasets(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets.
    Uses full datasets with all features for post-training bias analysis.
    
    Args:
        data_dir: Directory containing prepared datasets
        
    Returns:
        Tuple of (train_df, val_df, test_df) with all features
    """
    print(f"Loading datasets from: {data_dir}")
    
    # Load full datasets
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print(f"Train: {len(train_df)} samples with {len(train_df.columns)} features")
    print(f"Validation: {len(val_df)} samples with {len(val_df.columns)} features")
    print(f"Test: {len(test_df)} samples with {len(test_df.columns)} features")
    print("\nNote: BioBART uses only input_text and target_summary for training")
    print("Other features retained for bias detection and model evaluation")
    
    return train_df, val_df, test_df


def analyze_token_lengths(df: pd.DataFrame, tokenizer, max_input: int, max_target: int):
    """
    Analyze token lengths in the dataset to verify configuration.
    Helps ensure max_length parameters are appropriate for the data.
    """
    print("\n" + "="*60)
    print("ANALYZING TOKEN LENGTHS")
    print("="*60)
    
    # Sample for speed (analyze first 100 records)
    sample_size = min(100, len(df))
    sample_df = df.head(sample_size)
    
    input_lengths = []
    target_lengths = []
    
    # Tokenize samples to get actual token counts
    for _, row in sample_df.iterrows():
        input_tokens = tokenizer.encode(str(row['input_text']), truncation=True, max_length=max_input)
        target_tokens = tokenizer.encode(str(row['target_summary']), truncation=True, max_length=max_target)
        input_lengths.append(len(input_tokens))
        target_lengths.append(len(target_tokens))
    
    input_lengths = np.array(input_lengths)
    target_lengths = np.array(target_lengths)
    
    # Display input statistics
    print(f"Input tokens (n={sample_size}):")
    print(f"  Mean: {input_lengths.mean():.0f}, Median: {np.median(input_lengths):.0f}")
    print(f"  Max: {input_lengths.max()}, Min: {input_lengths.min()}")
    print(f"  Truncated: {(input_lengths >= max_input).sum()} ({(input_lengths >= max_input).sum()/sample_size*100:.1f}%)")
    
    # Display target statistics
    print(f"\nTarget tokens (n={sample_size}):")
    print(f"  Mean: {target_lengths.mean():.0f}, Median: {np.median(target_lengths):.0f}")
    print(f"  Max: {target_lengths.max()}, Min: {target_lengths.min()}")
    print(f"  Truncated: {(target_lengths >= max_target).sum()} ({(target_lengths >= max_target).sum()/sample_size*100:.1f}%)")
    
    # Warning if many targets are being truncated
    if target_lengths.max() > max_target:
        print(f"\nWarning: Some targets exceed {max_target} tokens")
        print(f"Consider increasing MAX_TARGET_LENGTH to {int(target_lengths.max() * 1.1)}")


def compute_metrics(eval_pred, tokenizer):
    """
    Compute ROUGE metrics for model evaluation.
    Uses sampling during training for faster computation.
    Full metrics are computed during final evaluation.
    """
    predictions, labels = eval_pred
    
    # Decode predictions from token IDs to text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (these are padding tokens that should be ignored)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Sample for faster computation during training
    # Full evaluation is done at the end
    sample_size = min(50, len(decoded_preds))
    decoded_preds_sample = decoded_preds[:sample_size]
    decoded_labels_sample = decoded_labels[:sample_size]
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compute ROUGE for each prediction-reference pair
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, label in zip(decoded_preds_sample, decoded_labels_sample):
        # Skip empty predictions or references
        if not pred.strip() or not label.strip():
            continue
        scores = scorer.score(label, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # Return average scores
    return {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0
    }


def evaluate_full_dataset(trainer, dataset, dataset_name="test"):
    """
    Run comprehensive evaluation on a full dataset.
    Used for final validation and test set evaluation.
    """
    print(f"\n{'='*60}")
    print(f"FULL EVALUATION ON {dataset_name.upper()} SET")
    print(f"{'='*60}")
    
    # Run evaluation
    metrics = trainer.evaluate(eval_dataset=dataset)
    
    # Display results
    print(f"{dataset_name.capitalize()} Metrics:")
    print(f"  ROUGE-1: {metrics.get('eval_rouge1', 0):.4f}")
    print(f"  ROUGE-2: {metrics.get('eval_rouge2', 0):.4f}")
    print(f"  ROUGE-L: {metrics.get('eval_rougeL', 0):.4f}")
    print(f"  Loss: {metrics.get('eval_loss', 0):.4f}")
    
    return metrics


def generate_diverse_samples(model, tokenizer, test_df, device, n_samples=10):
    """
    Generate summaries for diverse test samples.
    Attempts to sample across different demographics for comprehensive evaluation.
    """
    print(f"\n{'='*60}")
    print("GENERATING DIVERSE SAMPLE SUMMARIES")
    print(f"{'='*60}")
    
    generated_samples = []
    
    # Try to get diverse samples across demographics
    if 'age_group' in test_df.columns and 'gender' in test_df.columns:
        # Sample one from each demographic group
        sampled = test_df.groupby(['age_group', 'gender']).head(1).head(n_samples)
    else:
        # Use first n samples if demographics not available
        sampled = test_df.head(n_samples)
    
    print(f"Generating {len(sampled)} summaries...")
    
    # Generate summary for each sample
    for idx, (_, row) in enumerate(sampled.iterrows(), 1):
        input_text = str(row['input_text'])
        reference_summary = str(row['target_summary'])
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Generate summary using beam search
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode generated tokens to text
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display sample
        print(f"\n--- Sample {idx} ---")
        if 'age_group' in row and 'gender' in row:
            print(f"Demographics: {row.get('age_group', 'N/A')}, {row.get('gender', 'N/A')}")
        print(f"Input length: {len(input_text)} chars")
        print(f"Reference: {reference_summary[:150]}...")
        print(f"Generated: {generated_summary[:150]}...")
        
        # Store sample
        generated_samples.append({
            'hadm_id': int(row['hadm_id']) if 'hadm_id' in row else idx,
            'age_group': str(row.get('age_group', 'Unknown')),
            'gender': str(row.get('gender', 'Unknown')),
            'input_length': len(input_text),
            'reference_summary': reference_summary,
            'generated_summary': generated_summary
        })
    
    return generated_samples


def train_biobart_summarization(
    data_dir: str,
    output_dir: str,
    config: Dict = None,
    use_gpu: bool = False
):
    """
    Fine-tune BioBART model for medical text summarization.
    
    Complete training pipeline:
    1. Load and tokenize datasets
    2. Initialize BioBART model
    3. Configure training parameters
    4. Train with validation monitoring
    5. Track experiments with MLflow
    6. Evaluate on validation and test sets
    7. Save best model
    8. Generate diverse evaluation samples
    
    Args:
        data_dir: Directory with prepared train/val/test datasets
        output_dir: Directory to save trained model and results
        config: Model configuration dictionary
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Dictionary with model path, metrics, and MLflow run ID
    """
    
    print("="*60)
    print("BIOBART FINE-TUNING FOR MEDICAL SUMMARIZATION")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Determine device (CPU or GPU)
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    
    if device == 'cpu':
        print("Training on CPU (Mac). This will take 2-3 hours.")
        print("Using reduced batch size and gradient accumulation for stability")
    else:
        print("Training on GPU. This will take 30-45 minutes.")
    
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Initialize MLflow experiment tracking
    mlflow_dir = os.path.join(output_dir, '..', 'logs', 'mlruns')
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")
    mlflow.set_experiment("biobart_mimic_summarization")
    
    print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"biobart_finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log hyperparameters to MLflow
        print("\nLogging hyperparameters to MLflow...")
        mlflow.log_params({
            'model_name': MODEL_NAME,
            'max_input_length': MAX_INPUT_LENGTH,
            'max_target_length': MAX_TARGET_LENGTH,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'effective_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'warmup_steps': WARMUP_STEPS,
            'eval_steps': EVAL_STEPS,
            'device': device,
            'random_seed': RANDOM_SEED
        })
        
        # Load datasets
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        train_df, val_df, test_df = load_datasets(data_dir)
        
        # Log dataset statistics
        mlflow.log_params({
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'total_samples': len(train_df) + len(val_df) + len(test_df)
        })
        
        # Initialize BioBART tokenizer
        print("\n" + "="*60)
        print("INITIALIZING BIOBART TOKENIZER")
        print("="*60)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Tokenizer loaded: {MODEL_NAME}")
        print(f"Vocabulary size: {len(tokenizer)}")
        
        # Analyze token lengths in the data
        analyze_token_lengths(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
        
        # Load BioBART model for sequence-to-sequence generation
        print("\n" + "="*60)
        print("LOADING BIOBART MODEL")
        print("="*60)
        print(f"Model: {MODEL_NAME}")
        print("Pre-training: PubMed biomedical literature")
        print("Architecture: Encoder-Decoder (BART)")
        print("Task: Abstractive summarization")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Move model to device (CPU or GPU)
        model = model.to(device)
        print(f"Model loaded and moved to {device.upper()}")
        
        # Create PyTorch datasets
        print("\n" + "="*60)
        print("CREATING PYTORCH DATASETS")
        print("="*60)
        train_dataset = MIMICSummarizationDataset(
            train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
        )
        val_dataset = MIMICSummarizationDataset(
            val_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
        )
        test_dataset = MIMICSummarizationDataset(
            test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
        )
        
        # Configure training arguments
        # These parameters control how the model is trained
        print("\n" + "="*60)
        print("CONFIGURING TRAINING ARGUMENTS")
        print("="*60)
        
        training_args = Seq2SeqTrainingArguments(
            # Output and logging directories
            output_dir=output_dir,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=50,
            
            # Evaluation and saving strategy
            evaluation_strategy="steps",  # Evaluate every N steps
            eval_steps=EVAL_STEPS,  # Evaluate every 500 steps
            save_strategy="steps",  # Save checkpoint every N steps
            save_steps=EVAL_STEPS,  # Save every 500 steps
            save_total_limit=2,  # Keep only best 2 checkpoints
            
            # Training parameters
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Accumulate gradients
            num_train_epochs=NUM_EPOCHS,
            warmup_steps=WARMUP_STEPS,
            weight_decay=0.01,  # L2 regularization
            
            # Model selection criteria
            load_best_model_at_end=True,  # Load best checkpoint at end
            metric_for_best_model='rougeL',  # Use ROUGE-L for model selection
            greater_is_better=True,  # Higher ROUGE is better
            
            # Generation parameters for evaluation
            predict_with_generate=True,  # Generate summaries during eval
            generation_max_length=MAX_TARGET_LENGTH,  # Maximum tokens to generate
            generation_num_beams=2,  # Beam search width
            
            # Memory optimization
            eval_accumulation_steps=8,  # Accumulate eval batches before computing metrics
            
            # Performance optimization
            fp16=(device == 'cuda'),  # Mixed precision for GPU only
            dataloader_num_workers=0,  # Important for CPU training
            
            # Experiment tracking
            report_to=["mlflow"],  # Log to MLflow
        )
        
        print("Training configuration:")
        print(f"  Epochs: {NUM_EPOCHS}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Warmup steps: {WARMUP_STEPS}")
        print(f"  Eval steps: {EVAL_STEPS}")
        print(f"  Generation max length: {MAX_TARGET_LENGTH}")
        print(f"  Eval accumulation: 8 steps")
        
        # Initialize data collator
        # Handles batching and padding of variable-length sequences
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )
        
        # Initialize early stopping callback
        # Stops training if validation metric doesn't improve
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.0001
        )
        
        # Initialize Seq2Seq trainer
        # Handles training loop, evaluation, and logging
        print("\n" + "="*60)
        print("INITIALIZING TRAINER")
        print("="*60)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
            callbacks=[early_stopping]
        )
        
        print("Trainer initialized successfully")
        print(f"Training on {len(train_dataset)} samples")
        print(f"Validating on {len(val_dataset)} samples")
        print("Early stopping: patience=3, threshold=0.0001")
        
        # Start training
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)
        print(f"Estimated time: {'2-3 hours on CPU' if device == 'cpu' else '30-45 minutes on GPU'}")
        print("Progress will be logged every 50 steps")
        print("Evaluation every 500 steps")
        
        try:
            train_result = trainer.train()
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            print("Saving current model state...")
            trainer.save_model(os.path.join(output_dir, 'interrupted_checkpoint'))
            raise
        
        # Log training metrics to MLflow
        print("\n" + "="*60)
        print("TRAINING COMPLETE - LOGGING METRICS")
        print("="*60)
        
        mlflow.log_metrics({
            'train_loss': train_result.training_loss,
            'train_runtime_seconds': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second']
        })
        
        print(f"Training loss: {train_result.training_loss:.4f}")
        print(f"Training time: {train_result.metrics['train_runtime']:.1f} seconds")
        
        # Evaluate on validation set
        print("\n" + "="*60)
        print("EVALUATING ON VALIDATION SET")
        print("="*60)
        
        val_metrics = evaluate_full_dataset(trainer, val_dataset, "validation")
        
        # Log validation metrics to MLflow
        mlflow.log_metrics({
            'val_rouge1': val_metrics['eval_rouge1'],
            'val_rouge2': val_metrics['eval_rouge2'],
            'val_rougeL': val_metrics['eval_rougeL'],
            'val_loss': val_metrics['eval_loss']
        })
        
        # Evaluate on test set
        test_metrics = evaluate_full_dataset(trainer, test_dataset, "test")
        
        # Log test metrics to MLflow
        mlflow.log_metrics({
            'test_rouge1': test_metrics['eval_rouge1'],
            'test_rouge2': test_metrics['eval_rouge2'],
            'test_rougeL': test_metrics['eval_rougeL'],
            'test_loss': test_metrics['eval_loss']
        })
        
        # Save trained model and tokenizer
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to: {output_dir}")
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "biobart_model")
        
        # Generate diverse sample summaries for quality review
        generated_samples = generate_diverse_samples(
            model, tokenizer, test_df, device, n_samples=10
        )
        
        # Save generated samples
        samples_file = os.path.join(output_dir, 'generated_samples.json')
        with open(samples_file, 'w') as f:
            json.dump(generated_samples, f, indent=2)
        
        print(f"\nGenerated samples saved to: {samples_file}")
        
        # Display final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best model saved to: {output_dir}")
        print(f"\nValidation Metrics:")
        print(f"  ROUGE-1: {val_metrics['eval_rouge1']:.4f}")
        print(f"  ROUGE-2: {val_metrics['eval_rouge2']:.4f}")
        print(f"  ROUGE-L: {val_metrics['eval_rougeL']:.4f}")
        print(f"\nTest Metrics:")
        print(f"  ROUGE-1: {test_metrics['eval_rouge1']:.4f}")
        print(f"  ROUGE-2: {test_metrics['eval_rouge2']:.4f}")
        print(f"  ROUGE-L: {test_metrics['eval_rougeL']:.4f}")
        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"View results: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")
        print("="*60)
        
        return {
            'model_path': output_dir,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'mlflow_run_id': mlflow.active_run().info.run_id
        }


if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Fine-tune BioBART for medical discharge summary summarization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on CPU (Mac) with default settings
  python train_biobart.py
  
  # Train on GPU (GCP)
  python train_biobart.py --use-gpu
  
  # Use custom data directory
  python train_biobart.py --data-dir path/to/data
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='model-development/data/model_ready',
        help='Directory with prepared train/val/test data (default: %(default)s)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model-development/models/biobart_summarization',
        help='Directory to save trained model (default: %(default)s)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for training if available (for GCP deployment)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='model-development/configs/model_config.json',
        help='Path to model configuration file (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        print("Please run prepare_model_data.py first")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Train model
    print("\nStarting BioBART fine-tuning...\n")
    
    try:
        results = train_biobart_summarization(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            use_gpu=args.use_gpu
        )
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Model saved: {results['model_path']}")
        print(f"Val ROUGE-L: {results['val_metrics']['eval_rougeL']:.4f}")
        print(f"Test ROUGE-L: {results['test_metrics']['eval_rougeL']:.4f}")
        print(f"MLflow run: {results['mlflow_run_id']}")
        print("="*60)
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR DURING TRAINING")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print("\nIf training crashed, check:")
        print("  1. Sufficient disk space")
        print("  2. Memory available (close other apps)")
        print("  3. Data files exist and are valid")
        print("  4. Dependencies installed correctly")
        raise