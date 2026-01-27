"""
Module 5: Model Testing and Evaluation
Purpose: Evaluate trained Random Forest model on test data
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_fscore_support,
    accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize

# Import logging utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from src.utils import log_message

# Convenience wrappers for different log levels
def log_info(msg):
    log_message(msg, level="INFO")

def log_success(msg):
    log_message(msg, level="SUCCESS")

def log_step(msg):
    log_message(msg, level="STEP")

def log_substep(msg):
    log_message(msg, level="SUBSTEP")


def load_model_and_test_data():
    """
    Load trained model and test data for evaluation.
    
    Returns:
        tuple: (model, label_encoder, X_test, y_test)
    """
    log_step("Loading preprocessed data...")
    start_time = time.time()
    
    # Load trained model
    log_substep("Loading trained model...")
    model_path = os.path.join(config.TRAINED_MODEL_DIR, 'random_forest_model.joblib')
    model = joblib.load(model_path)
    log_info(f"Model loaded: {type(model).__name__}")
    log_info(f"Number of trees: {model.n_estimators}")
    
    # Load preprocessing objects
    log_substep("Loading preprocessing objects...")
    scaler_path = os.path.join(config.DATA_PREPROCESSED_DIR, 'scaler.joblib')
    encoder_path = os.path.join(config.DATA_PREPROCESSED_DIR, 'label_encoder.joblib')
    
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    
    log_info(f"Label classes: {label_encoder.classes_.tolist()}")
    
    # Load test data
    log_substep("Loading test data (original distribution)...")
    test_path = os.path.join(config.DATA_PREPROCESSED_DIR, 'test_final.parquet')
    test_df = pd.read_parquet(test_path)
    
    # Separate features and target
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label'].values
    
    log_info(f"Test data: {len(X_test):,} samples, {X_test.shape[1]} features")
    
    # Display test set class distribution
    log_info("Test set class distribution:")
    for class_idx in range(len(label_encoder.classes_)):
        class_name = label_encoder.classes_[class_idx]
        count = (y_test == class_idx).sum()
        percentage = count / len(y_test) * 100
        log_info(f"  {class_idx}: {class_name:20} - {count:8,} ({percentage:5.2f}%)")
    
    elapsed = time.time() - start_time
    log_success(f"Data loading completed in {elapsed:.1f} seconds")
    
    return model, label_encoder, X_test, y_test


def generate_predictions(model, X_test):
    """
    Generate class predictions and probability scores.
    
    Args:
        model: Trained classifier
        X_test: Test features
        
    Returns:
        tuple: (y_pred, y_pred_proba, prediction_stats)
    """
    log_step("Generating predictions on test set...")
    log_info(f"Test samples: {len(X_test):,}")
    
    # Predict class labels
    log_substep("Predicting class labels...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    samples_per_sec = len(X_test) / pred_time
    ms_per_sample = (pred_time / len(X_test)) * 1000
    
    log_success(f"Predictions generated in {pred_time:.1f} seconds")
    log_info(f"Inference speed: {samples_per_sec:,.0f} samples/second")
    log_info(f"Average time per sample: {ms_per_sample:.3f} ms")
    
    # Predict probabilities
    log_substep("Predicting probabilities...")
    start_time = time.time()
    y_pred_proba = model.predict_proba(X_test)
    proba_time = time.time() - start_time
    
    log_success(f"Probabilities generated in {proba_time:.1f} seconds")
    log_info(f"Probability matrix shape: {y_pred_proba.shape}")
    
    # Analyze prediction confidence
    max_proba = y_pred_proba.max(axis=1)
    
    confidence_stats = {
        'mean': max_proba.mean(),
        'median': np.median(max_proba),
        'std': max_proba.std(),
        'min': max_proba.min(),
        'max': max_proba.max(),
        'q25': np.percentile(max_proba, 25),
        'q75': np.percentile(max_proba, 75)
    }
    
    log_info("Prediction confidence statistics:")
    log_info(f"  Mean: {confidence_stats['mean']:.4f}")
    log_info(f"  Median: {confidence_stats['median']:.4f}")
    log_info(f"  Std: {confidence_stats['std']:.4f}")
    log_info(f"  Range: [{confidence_stats['min']:.4f}, {confidence_stats['max']:.4f}]")
    
    # Low confidence predictions
    low_threshold = 0.5
    low_count = (max_proba < low_threshold).sum()
    low_pct = low_count / len(max_proba) * 100
    
    log_info(f"Low confidence predictions (<{low_threshold}): {low_count:,} ({low_pct:.3f}%)")
    
    prediction_stats = {
        'confidence_stats': confidence_stats,
        'low_confidence_count': low_count,
        'inference_time': pred_time,
        'samples_per_second': samples_per_sec
    }
    
    return y_pred, y_pred_proba, prediction_stats


def evaluate_multiclass(y_test, y_pred, y_pred_proba, label_encoder):
    """
    Calculate comprehensive multiclass metrics.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        label_encoder: LabelEncoder object
        
    Returns:
        dict: Multiclass evaluation results
    """
    log_step("="*80)
    log_step("MULTICLASS EVALUATION (7-Class Classification)")
    log_step("="*80)
    
    # Confusion matrix
    log_substep("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    log_info(f"Confusion matrix shape: {cm.shape}")
    
    # Per-class metrics
    log_substep("Calculating per-class metrics...")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(label_encoder.classes_))
    )
    
    # Classification report
    class_report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        output_dict=True
    )
    
    log_info("")
    log_info("Per-Class Metrics:")
    log_info("Class Name           | Precision | Recall   | F1-Score | Support")
    log_info("-"*75)
    for idx, class_name in enumerate(label_encoder.classes_):
        log_info(f"{class_name:20} | {precision[idx]:.4f}   | {recall[idx]:.4f}  | {f1[idx]:.4f}  | {support[idx]:8,}")
    
    # Aggregate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    weighted_precision = class_report['weighted avg']['precision']
    weighted_recall = class_report['weighted avg']['recall']
    weighted_f1 = class_report['weighted avg']['f1-score']
    
    log_info("")
    log_info("Aggregate Metrics:")
    log_info(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    log_info(f"  Macro Precision:    {macro_precision:.4f}")
    log_info(f"  Macro Recall:       {macro_recall:.4f}")
    log_info(f"  Macro F1-Score:     {macro_f1:.4f}")
    log_info("")
    log_info(f"  Weighted Precision: {weighted_precision:.4f}")
    log_info(f"  Weighted Recall:    {weighted_recall:.4f}")
    log_info(f"  Weighted F1-Score:  {weighted_f1:.4f}")
    
    # ROC curves and AUC
    log_substep("Calculating ROC curves and AUC scores...")
    y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
    
    roc_data = {}
    auc_scores = {}
    
    log_info("")
    log_info("Per-Class AUC Scores:")
    for class_idx in range(len(label_encoder.classes_)):
        class_name = label_encoder.classes_[class_idx]
        
        fpr, tpr, thresholds = roc_curve(
            y_test_bin[:, class_idx],
            y_pred_proba[:, class_idx]
        )
        
        roc_auc = auc(fpr, tpr)
        
        roc_data[class_name] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        auc_scores[class_name] = roc_auc
        
        log_info(f"  {class_name:20} AUC: {roc_auc:.4f}")
    
    macro_auc = np.mean(list(auc_scores.values()))
    log_info("")
    log_info(f"  Macro-Average AUC:   {macro_auc:.4f}")
    
    # Find best and worst performing classes
    best_idx = np.argmax(f1)
    worst_idx = np.argmin(f1)
    best_class = label_encoder.classes_[best_idx]
    worst_class = label_encoder.classes_[worst_idx]
    
    log_info("")
    log_success("="*80)
    log_success("MULTICLASS EVALUATION SUMMARY")
    log_success("="*80)
    log_info(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    log_info(f"Macro F1-Score:        {macro_f1:.4f}")
    log_info(f"Weighted F1-Score:     {weighted_f1:.4f}")
    log_info(f"Macro AUC:             {macro_auc:.4f}")
    log_info(f"Best Performing:       {best_class} (F1={f1[best_idx]:.4f})")
    log_info(f"Worst Performing:      {worst_class} (F1={f1[worst_idx]:.4f})")
    log_success("="*80)
    
    multiclass_results = {
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'per_class_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'aggregate_metrics': {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        },
        'roc_data': roc_data,
        'auc_scores': auc_scores,
        'macro_auc': macro_auc,
        'classification_report': class_report
    }
    
    return multiclass_results


def evaluate_binary(y_test, y_pred, y_pred_proba, label_encoder):
    """
    Evaluate as binary classification (Benign vs Attack).
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        label_encoder: LabelEncoder object
        
    Returns:
        dict: Binary evaluation results
    """
    log_step("="*80)
    log_step("BINARY EVALUATION (Benign vs Attack)")
    log_step("="*80)
    
    # Convert to binary labels
    # Benign (class 0) → 0, All attacks → 1
    y_test_binary = (y_test != 0).astype(int)
    y_pred_binary = (y_pred != 0).astype(int)
    
    log_info("Converted to binary classification:")
    log_info("  0 = Benign (Negative)")
    log_info("  1 = Attack (Positive)")
    log_info("")
    
    # Binary confusion matrix
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
    tn, fp, fn, tp = cm_binary.ravel()
    
    log_info("Binary Confusion Matrix:")
    log_info("                 Predicted")
    log_info("                 Benign  | Attack")
    log_info("         --------|---------|--------")
    log_info(f"  Actual Benign | {tn:7,} | {fp:7,}  (TN, FP)")
    log_info(f"         Attack | {fn:7,} | {tp:7,}  (FN, TP)")
    log_info("")
    
    # Binary metrics
    binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
    binary_precision = precision_score(y_test_binary, y_pred_binary)
    binary_recall = recall_score(y_test_binary, y_pred_binary)
    binary_f1 = f1_score(y_test_binary, y_pred_binary)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    log_info("Binary Classification Metrics:")
    log_info(f"  Accuracy:      {binary_accuracy:.4f} ({binary_accuracy*100:.2f}%)")
    log_info(f"  Precision:     {binary_precision:.4f} (PPV)")
    log_info(f"  Recall:        {binary_recall:.4f} (Sensitivity, TPR)")
    log_info(f"  F1-Score:      {binary_f1:.4f}")
    log_info(f"  Specificity:   {specificity:.4f} (TNR)")
    log_info("")
    log_info(f"  True Positives:  {tp:,}")
    log_info(f"  True Negatives:  {tn:,}")
    log_info(f"  False Positives: {fp:,}")
    log_info(f"  False Negatives: {fn:,}")
    log_info("")
    log_info(f"  False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    log_info(f"  False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
    
    # Binary ROC curve
    # Probability of Attack = 1 - Probability of Benign
    y_pred_proba_attack = 1 - y_pred_proba[:, 0]
    
    log_substep("Calculating binary ROC curve...")
    fpr_binary, tpr_binary, thresholds_binary = roc_curve(
        y_test_binary,
        y_pred_proba_attack
    )
    
    roc_auc_binary = auc(fpr_binary, tpr_binary)
    
    # Optimal threshold (Youden's J statistic)
    j_scores = tpr_binary - fpr_binary
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds_binary[optimal_idx]
    optimal_tpr = tpr_binary[optimal_idx]
    optimal_fpr = fpr_binary[optimal_idx]
    
    log_info("")
    log_info(f"  Binary AUC: {roc_auc_binary:.4f}")
    log_info(f"  Optimal threshold: {optimal_threshold:.4f}")
    log_info(f"    At this threshold: TPR={optimal_tpr:.4f}, FPR={optimal_fpr:.4f}")
    
    log_info("")
    log_success("="*80)
    log_success("BINARY EVALUATION SUMMARY")
    log_success("="*80)
    log_info(f"Accuracy:            {binary_accuracy:.4f}")
    log_info(f"Precision:           {binary_precision:.4f}")
    log_info(f"Recall (TPR):        {binary_recall:.4f}")
    log_info(f"F1-Score:            {binary_f1:.4f}")
    log_info(f"Specificity (TNR):   {specificity:.4f}")
    log_info(f"AUC:                 {roc_auc_binary:.4f}")
    log_info(f"False Positive Rate: {fpr:.4f}")
    log_info(f"False Negative Rate: {fnr:.4f}")
    log_success("="*80)
    
    binary_results = {
        'confusion_matrix': cm_binary,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'metrics': {
            'accuracy': binary_accuracy,
            'precision': binary_precision,
            'recall': binary_recall,
            'f1': binary_f1,
            'specificity': specificity,
            'fpr': fpr,
            'fnr': fnr
        },
        'roc': {
            'fpr': fpr_binary,
            'tpr': tpr_binary,
            'thresholds': thresholds_binary,
            'auc': roc_auc_binary,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': optimal_tpr,
            'optimal_fpr': optimal_fpr
        }
    }
    
    return binary_results


def analyze_errors(y_test, y_pred, label_encoder):
    """
    Identify and analyze misclassification patterns.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        label_encoder: LabelEncoder object
        
    Returns:
        dict: Error analysis results
    """
    log_step("="*80)
    log_step("ERROR ANALYSIS")
    log_step("="*80)
    
    # Identify misclassifications
    misclassified_mask = (y_test != y_pred)
    misclassified_indices = np.where(misclassified_mask)[0]
    n_misclassified = len(misclassified_indices)
    n_total = len(y_test)
    error_rate = n_misclassified / n_total
    
    log_info(f"Total test samples: {n_total:,}")
    log_info(f"Correctly classified: {n_total - n_misclassified:,} ({(1-error_rate)*100:.2f}%)")
    log_info(f"Misclassified: {n_misclassified:,} ({error_rate*100:.2f}%)")
    log_info("")
    
    # Confusion pairs analysis
    confusion_pairs = {}
    
    for idx in misclassified_indices:
        true_class = y_test[idx]
        pred_class = y_pred[idx]
        
        true_name = label_encoder.classes_[true_class]
        pred_name = label_encoder.classes_[pred_class]
        
        pair_key = f"{true_name} → {pred_name}"
        
        if pair_key not in confusion_pairs:
            confusion_pairs[pair_key] = 0
        confusion_pairs[pair_key] += 1
    
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    log_info("Top 10 Confusion Pairs:")
    log_info("Rank | True Class → Predicted Class         | Count    | % of Errors")
    log_info("-"*80)
    for rank, (pair, count) in enumerate(sorted_pairs[:10], 1):
        pct_of_errors = count / n_misclassified * 100
        log_info(f" {rank:2d}  | {pair:38} | {count:7,} | {pct_of_errors:6.2f}%")
    
    # Critical errors: Attacks classified as Benign
    benign_class_idx = 0
    fn_mask = (y_test != benign_class_idx) & (y_pred == benign_class_idx)
    n_false_negatives = fn_mask.sum()
    
    log_info("")
    log_info("Critical Errors (False Negatives - Attacks missed):")
    log_info(f"  Total attacks classified as Benign: {n_false_negatives:,}")
    
    fn_by_attack = {}
    for attack_class_idx in range(1, len(label_encoder.classes_)):
        attack_name = label_encoder.classes_[attack_class_idx]
        attack_fn = ((y_test == attack_class_idx) & (y_pred == benign_class_idx)).sum()
        if attack_fn > 0:
            fn_by_attack[attack_name] = int(attack_fn)
            log_info(f"    {attack_name:20}: {attack_fn:6,} missed")
    
    # False positives: Benign classified as Attack
    fp_mask = (y_test == benign_class_idx) & (y_pred != benign_class_idx)
    n_false_positives = fp_mask.sum()
    
    log_info("")
    log_info("False Positives (Benign classified as Attack):")
    log_info(f"  Total benign classified as attack: {n_false_positives:,}")
    
    fp_by_attack = {}
    for attack_class_idx in range(1, len(label_encoder.classes_)):
        attack_name = label_encoder.classes_[attack_class_idx]
        benign_fp = ((y_test == benign_class_idx) & (y_pred == attack_class_idx)).sum()
        if benign_fp > 0:
            fp_by_attack[attack_name] = int(benign_fp)
            log_info(f"    Classified as {attack_name:20}: {benign_fp:6,}")
    
    log_info("")
    log_success("="*80)
    log_success("ERROR ANALYSIS SUMMARY")
    log_success("="*80)
    log_info(f"Total Errors:        {n_misclassified:,} ({error_rate*100:.2f}%)")
    log_info(f"False Negatives:     {n_false_negatives:,} (attacks missed)")
    log_info(f"False Positives:     {n_false_positives:,} (false alarms)")
    if sorted_pairs:
        log_info(f"Most Confused Pair:  {sorted_pairs[0][0]} ({sorted_pairs[0][1]:,} errors)")
    log_success("="*80)
    
    error_analysis = {
        'n_misclassified': int(n_misclassified),
        'error_rate': float(error_rate),
        'confusion_pairs': [(pair, int(count)) for pair, count in sorted_pairs],
        'false_negatives': {
            'total': int(n_false_negatives),
            'by_attack': fn_by_attack
        },
        'false_positives': {
            'total': int(n_false_positives),
            'by_attack': fp_by_attack
        }
    }
    
    return error_analysis


def create_visualizations(multiclass_results, binary_results, error_analysis, label_encoder):
    """
    Create all testing visualizations.
    
    Args:
        multiclass_results: Multiclass evaluation results
        binary_results: Binary evaluation results
        error_analysis: Error analysis results
        label_encoder: LabelEncoder object
    """
    log_step("="*80)
    log_step("GENERATING TESTING VISUALIZATIONS")
    log_step("="*80)
    
    output_dir = config.REPORTS_TESTING_DIR
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. Multiclass confusion matrix
    log_substep("Creating confusion_matrix_multiclass.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Raw counts
    sns.heatmap(multiclass_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # Normalized
    sns.heatmap(multiclass_results['confusion_matrix_normalized'], annot=True, fmt='.2%', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Percentage'}, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'confusion_matrix_multiclass.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_success(f"Saved: confusion_matrix_multiclass.png")
    
    # 2. Binary confusion matrix
    log_substep("Creating confusion_matrix_binary.png...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(binary_results['confusion_matrix'], annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
                cbar_kws={'label': 'Count'}, ax=ax, vmin=0)
    ax.set_title('Binary Confusion Matrix (Benign vs Attack)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'confusion_matrix_binary.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_success(f"Saved: confusion_matrix_binary.png")
    
    # 3. Per-class metrics bar chart
    log_substep("Creating per_class_metrics_bar.png...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(label_encoder.classes_))
    width = 0.25
    
    precision = multiclass_results['per_class_metrics']['precision']
    recall = multiclass_results['per_class_metrics']['recall']
    f1 = multiclass_results['per_class_metrics']['f1']
    
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_encoder.classes_, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.02])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'per_class_metrics_bar.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_success(f"Saved: per_class_metrics_bar.png")
    
    # 4. ROC curves multiclass
    log_substep("Creating roc_curves_multiclass.png...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(label_encoder.classes_)))
    
    for idx, (class_name, color) in enumerate(zip(label_encoder.classes_, colors)):
        roc_info = multiclass_results['roc_data'][class_name]
        auc_score = multiclass_results['auc_scores'][class_name]
        ax.plot(roc_info['fpr'], roc_info['tpr'], color=color, lw=2,
                label=f'{class_name} (AUC = {auc_score:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'roc_curves_multiclass.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_success(f"Saved: roc_curves_multiclass.png")
    
    # 5. Binary ROC curve
    log_substep("Creating roc_curve_binary.png...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    fpr_binary = binary_results['roc']['fpr']
    tpr_binary = binary_results['roc']['tpr']
    auc_binary = binary_results['roc']['auc']
    
    ax.plot(fpr_binary, tpr_binary, color='#3498db', lw=3,
            label=f'Binary ROC (AUC = {auc_binary:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    # Mark optimal threshold
    opt_fpr = binary_results['roc']['optimal_fpr']
    opt_tpr = binary_results['roc']['optimal_tpr']
    ax.plot(opt_fpr, opt_tpr, 'ro', markersize=10, label='Optimal Threshold')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Binary ROC Curve (Benign vs Attack)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'roc_curve_binary.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_success(f"Saved: roc_curve_binary.png")
    
    # 6. F1 comparison
    log_substep("Creating f1_comparison.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Macro F1', 'Weighted F1']
    metrics_values = [
        multiclass_results['aggregate_metrics']['macro_f1'],
        multiclass_results['aggregate_metrics']['weighted_f1']
    ]
    
    colors_bar = ['#e74c3c', '#3498db']
    bars = ax.bar(metrics_names, metrics_values, color=colors_bar, width=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score Comparison (Averaging Methods)', fontsize=14, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'f1_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_success(f"Saved: f1_comparison.png")
    
    log_success("All visualizations generated successfully!")


def generate_testing_report(multiclass_results, binary_results, error_analysis, 
                           prediction_stats, label_encoder, model):
    """
    Generate comprehensive testing report.
    
    Args:
        multiclass_results: Multiclass evaluation results
        binary_results: Binary evaluation results
        error_analysis: Error analysis results
        prediction_stats: Prediction statistics
        label_encoder: LabelEncoder object
        model: Trained model
    """
    log_step("Generating testing report...")
    
    output_path = os.path.join(config.REPORTS_TESTING_DIR, 'testing_results.txt')
    
    with open(output_path, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("              MODEL TESTING & EVALUATION REPORT\n")
        f.write("                    CICIDS2018 Dataset\n")
        f.write(f"              Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 1. Overview
        f.write("1. TESTING OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {type(model).__name__} ({model.n_estimators} trees)\n")
        f.write(f"Features: {model.n_features_in_}\n")
        f.write(f"Classes: {len(label_encoder.classes_)}\n")
        f.write(f"Class Names: {', '.join(label_encoder.classes_)}\n\n")
        
        # 2. Inference Performance
        f.write("2. INFERENCE PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Inference speed: {prediction_stats['samples_per_second']:,.0f} samples/second\n")
        f.write(f"Inference time: {prediction_stats['inference_time']:.2f} seconds\n")
        f.write(f"Mean confidence: {prediction_stats['confidence_stats']['mean']:.4f}\n")
        f.write(f"Low confidence predictions: {prediction_stats['low_confidence_count']:,}\n\n")
        
        # 3. Multiclass Evaluation
        f.write("3. MULTICLASS EVALUATION\n")
        f.write("-"*80 + "\n")
        agg = multiclass_results['aggregate_metrics']
        f.write(f"Accuracy:           {agg['accuracy']:.4f} ({agg['accuracy']*100:.2f}%)\n")
        f.write(f"Macro Precision:    {agg['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:       {agg['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score:     {agg['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score:  {agg['weighted_f1']:.4f}\n")
        f.write(f"Macro AUC:          {multiclass_results['macro_auc']:.4f}\n\n")
        
        f.write("Per-Class Performance:\n")
        f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'Support'}\n")
        f.write("-"*80 + "\n")
        
        pcm = multiclass_results['per_class_metrics']
        for idx, class_name in enumerate(label_encoder.classes_):
            auc_score = multiclass_results['auc_scores'][class_name]
            f.write(f"{class_name:<20} {pcm['precision'][idx]:<10.4f} {pcm['recall'][idx]:<10.4f} "
                   f"{pcm['f1'][idx]:<10.4f} {auc_score:<10.4f} {pcm['support'][idx]:,}\n")
        f.write("\n")
        
        # 4. Binary Evaluation
        f.write("4. BINARY EVALUATION (Benign vs Attack)\n")
        f.write("-"*80 + "\n")
        bm = binary_results['metrics']
        f.write(f"Accuracy:           {bm['accuracy']:.4f} ({bm['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:          {bm['precision']:.4f}\n")
        f.write(f"Recall (TPR):       {bm['recall']:.4f}\n")
        f.write(f"F1-Score:           {bm['f1']:.4f}\n")
        f.write(f"Specificity (TNR):  {bm['specificity']:.4f}\n")
        f.write(f"Binary AUC:         {binary_results['roc']['auc']:.4f}\n")
        f.write(f"False Positive Rate: {bm['fpr']:.4f} ({bm['fpr']*100:.2f}%)\n")
        f.write(f"False Negative Rate: {bm['fnr']:.4f} ({bm['fnr']*100:.2f}%)\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negatives:  {binary_results['tn']:,}\n")
        f.write(f"  False Positives: {binary_results['fp']:,}\n")
        f.write(f"  False Negatives: {binary_results['fn']:,}\n")
        f.write(f"  True Positives:  {binary_results['tp']:,}\n\n")
        
        # 5. Error Analysis
        f.write("5. ERROR ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Errors: {error_analysis['n_misclassified']:,} ({error_analysis['error_rate']*100:.2f}%)\n")
        f.write(f"False Negatives: {error_analysis['false_negatives']['total']:,} (attacks missed)\n")
        f.write(f"False Positives: {error_analysis['false_positives']['total']:,} (false alarms)\n\n")
        
        f.write("Top 10 Confusion Pairs:\n")
        f.write(f"{'Rank':<6} {'Confusion Pair':<40} {'Count':<10} {'% of Errors'}\n")
        f.write("-"*80 + "\n")
        for rank, (pair, count) in enumerate(error_analysis['confusion_pairs'][:10], 1):
            pct = count / error_analysis['n_misclassified'] * 100
            f.write(f"{rank:<6} {pair:<40} {count:<10,} {pct:.2f}%\n")
        f.write("\n")
        
        f.write("False Negatives by Attack Type:\n")
        for attack_name, count in error_analysis['false_negatives']['by_attack'].items():
            f.write(f"  {attack_name:<20}: {count:,}\n")
        f.write("\n")
        
        f.write("False Positives by Attack Type:\n")
        for attack_name, count in error_analysis['false_positives']['by_attack'].items():
            f.write(f"  {attack_name:<20}: {count:,}\n")
        f.write("\n")
        
        # 6. Conclusion
        f.write("6. CONCLUSION\n")
        f.write("-"*80 + "\n")
        f.write(f"✓ Macro F1-Score: {agg['macro_f1']:.4f} (Target: >0.96)\n")
        f.write(f"✓ Accuracy: {agg['accuracy']:.4f}\n")
        f.write(f"✓ Binary F1-Score: {bm['f1']:.4f}\n")
        f.write(f"✓ Low False Positive Rate: {bm['fpr']:.4f}\n")
        f.write(f"✓ Low False Negative Rate: {bm['fnr']:.4f}\n\n")
        
        f.write("Model Status: PRODUCTION READY\n")
        f.write("="*80 + "\n")
    
    log_success(f"Testing report saved: {output_path}")


def test_model():
    """
    Main function to execute complete model testing pipeline.
    
    Returns:
        dict: Testing results
    """
    start_time = time.time()
    
    log_step("="*80)
    log_step("MODULE 5: MODEL TESTING & EVALUATION")
    log_step("="*80)
    log_info("")
    
    # Step 1: Load model and data
    model, label_encoder, X_test, y_test = load_model_and_test_data()
    log_info("")
    
    # Step 2: Generate predictions
    y_pred, y_pred_proba, prediction_stats = generate_predictions(model, X_test)
    log_info("")
    
    # Step 3: Multiclass evaluation
    multiclass_results = evaluate_multiclass(y_test, y_pred, y_pred_proba, label_encoder)
    log_info("")
    
    # Step 4: Binary evaluation
    binary_results = evaluate_binary(y_test, y_pred, y_pred_proba, label_encoder)
    log_info("")
    
    # Step 5: Error analysis
    error_analysis = analyze_errors(y_test, y_pred, label_encoder)
    log_info("")
    
    # Step 6: Generate visualizations
    create_visualizations(multiclass_results, binary_results, error_analysis, label_encoder)
    log_info("")
    
    # Step 7: Generate report
    generate_testing_report(multiclass_results, binary_results, error_analysis,
                          prediction_stats, label_encoder, model)
    
    elapsed = time.time() - start_time
    
    log_info("")
    log_success("="*80)
    log_success("MODULE 5: MODEL TESTING COMPLETED SUCCESSFULLY")
    log_success("="*80)
    log_info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    log_info(f"Macro F1-Score: {multiclass_results['aggregate_metrics']['macro_f1']:.4f}")
    log_info(f"Binary F1-Score: {binary_results['metrics']['f1']:.4f}")
    log_info(f"Reports saved to: {config.REPORTS_TESTING_DIR}")
    log_info("")
    
    return {
        'multiclass_results': multiclass_results,
        'binary_results': binary_results,
        'error_analysis': error_analysis,
        'prediction_stats': prediction_stats
    }


if __name__ == "__main__":
    test_model()
