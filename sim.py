import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import logging

# Make the output pretty
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pca_svd(X, n_components):
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Apply SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # V contains the right singular vectors (principal components)
    components = Vt.T[:, :n_components]
    
    # Project data onto principal components
    X_pca = np.dot(X_centered, components)
    
    return X_pca, components, X_mean

def lda_fisher(X, y, n_components):
    # Get dimensions and class information
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    
    # Calculate overall mean
    mean_overall = np.mean(X, axis=0)
    
    # Calculate between-class scatter matrix
    S_B = np.zeros((n_features, n_features))
    for cls in classes:
        X_cls = X[y == cls]
        mean_cls = np.mean(X_cls, axis=0)
        n_samples_cls = X_cls.shape[0]
        
        mean_diff = (mean_cls - mean_overall).reshape(-1, 1)
        S_B += n_samples_cls * np.dot(mean_diff, mean_diff.T)
    
    # Calculate within-class scatter matrix
    S_W = np.zeros((n_features, n_features))
    for cls in classes:
        X_cls = X[y == cls]
        mean_cls = np.mean(X_cls, axis=0)
        
        centered_cls = X_cls - mean_cls
        S_W += np.dot(centered_cls.T, centered_cls)
    
    # Regularize S_W slightly to avoid singularity issues
    S_W += np.eye(n_features) * 1e-4
    
    # Solve eigenvalue problem for S_W^-1 * S_B
    try:
        S_W_inv = np.linalg.inv(S_W)
        eig_vals, eig_vecs = np.linalg.eigh(np.dot(S_W_inv, S_B))
    except np.linalg.LinAlgError:
        # If inversion fails, use pseudoinverse
        logger.warning("Using pseudoinverse for LDA computation")
        S_W_inv = np.linalg.pinv(S_W)
        eig_vals, eig_vecs = np.linalg.eigh(np.dot(S_W_inv, S_B))
        
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    
    # Select top n_components eigenvectors
    components = np.real(eig_vecs[:, :n_components])
    
    # Project data
    X_lda = np.dot(X, components)
    
    return X_lda, components

def get_lbp_value(image, i, j, radius=1, n_points=8): # Calculate LBP value for a pixel
    height, width = image.shape
    center = image[i, j]
    binary_pattern = 0
    
    for p in range(n_points):
        angle = 2 * np.pi * p / n_points
        x = j + radius * np.cos(angle)
        y = i - radius * np.sin(angle)
        
        # Bilinear interpolation for coordinates
        x1, y1 = int(np.floor(x)), int(np.floor(y))
        x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
        
        # Handle boundary conditions
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        tx, ty = x - x1, y - y1
        
        # Calculate interpolated value
        if 0 <= x1 < width and 0 <= y1 < height:
            top = (1 - tx) * image[y1, x1] + tx * image[y1, x2] if x2 < width else image[y1, x1]
            bottom = (1 - tx) * image[y2, x1] + tx * image[y2, x2] if y2 < height and x2 < width else image[y2, x1]
            neighbor = (1 - ty) * top + ty * bottom
        else:
            neighbor = 0
        
        # Update binary pattern
        if neighbor >= center:
            binary_pattern |= (1 << p)
    
    return binary_pattern

def lbph(images, radius=1, n_points=8, grid_x=8, grid_y=8):
    features = []
    
    for img_flat in images:
        # Reshape flat image to 2D if needed
        if len(img_flat.shape) == 1:
            # Assuming square images
            img_size = int(np.sqrt(img_flat.shape[0]))
            img = img_flat.reshape(img_size, img_size)
        else:
            img = img_flat
        
        height, width = img.shape
        
        # Initialize LBP image
        lbp_image = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate LBP for inner pixels (avoid borders for simplicity)
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                lbp_image[i, j] = get_lbp_value(img, i, j, radius, n_points)
        
        # Calculate histograms for grid cells
        cell_height = height // grid_y
        cell_width = width // grid_x
        histograms = []
        
        for i in range(grid_y):
            for j in range(grid_x):
                cell = lbp_image[i*cell_height:(i+1)*cell_height, 
                                j*cell_width:(j+1)*cell_width]
                hist, _ = np.histogram(cell, bins=range(257), density=True)
                histograms.append(hist)
        
        # Concatenate all histograms
        feature_vector = np.concatenate(histograms)
        features.append(feature_vector)
    
    return np.array(features)

def load_dataset(dataset_dir, scan_dir):
    # Load training data
    X_train = []
    y_train = []
    person_names = []
    
    logger.info(f"Loading training images from {dataset_dir}")
    
    for person_folder in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_folder)
        if os.path.isdir(person_path):
            person_names.append(person_folder)
            person_id = len(person_names) - 1
            logger.info(f"Loading images for {person_folder} (ID: {person_id})")
            
            image_count = 0
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        # Load image and convert to grayscale if needed
                        img = cv2.imread(img_path)
                        if img is None:
                            logger.warning(f"Failed to load image: {img_path}")
                            continue
                            
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
                        resized_img = cv2.resize(gray_img, (64, 64))  # Standard size
                        
                        X_train.append(resized_img.flatten())
                        y_train.append(person_id)
                        image_count += 1
                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {str(e)}")
            
            logger.info(f"  Loaded {image_count} images for {person_folder}")
    
    if not X_train:
        logger.error("No training images were found!")
        return None, None, None, None, None
    
    # Load test images
    X_test = []
    y_test = []
    
    logger.info(f"Loading test images from {scan_dir}")
    
    for img_file in os.listdir(scan_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Extract person name from filename
            person_name = img_file.split('.')[0].replace('_', ' ')
            
            if person_name in person_names:
                person_id = person_names.index(person_name)
                img_path = os.path.join(scan_dir, img_file)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.warning(f"Failed to load image: {img_path}")
                        continue
                        
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
                    resized_img = cv2.resize(gray_img, (64, 64))
                    
                    X_test.append(resized_img.flatten())
                    y_test.append(person_id)
                    logger.info(f"Loaded test image for {person_name}")
                except Exception as e:
                    logger.error(f"Error processing scan image {img_path}: {str(e)}")
            else:
                logger.warning(f"Unknown person in scan: {person_name}")
    
    if not X_test:
        logger.warning("No test images were found!")
        # If no test images, split training data
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X_train), np.array(y_train), test_size=0.2, stratify=np.array(y_train)
        )
        logger.info("Split training data for testing (80/20 split)")
        return X_train, X_test, y_train, y_test, person_names
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), person_names

def evaluate_classifier(X_train, X_test, y_train, y_test, person_names):
    # Initialize results container
    results = {}
    n_classes = len(person_names)
    
    # Define feature extraction parameters
    n_components_pca = min(50, X_train.shape[0] - 1)
    n_components_lda = min(n_classes - 1, X_train.shape[1])
    k_neighbors = 5
    
    # Apply PCA
    logger.info("Applying PCA...")
    X_train_pca, pca_components, pca_mean = pca_svd(X_train, n_components_pca)
    X_test_pca = np.dot(X_test - pca_mean, pca_components)
    
    # Apply LDA
    logger.info("Applying LDA...")
    X_train_lda, lda_components = lda_fisher(X_train, y_train, n_components_lda)
    X_test_lda = np.dot(X_test, lda_components)
    
    # Apply LBP
    logger.info("Applying LBP...")
    # Reshape flattened images back to 2D for LBP
    img_size = int(np.sqrt(X_train.shape[1]))
    X_train_reshaped = [img.reshape(img_size, img_size) for img in X_train]
    X_test_reshaped = [img.reshape(img_size, img_size) for img in X_test]
    
    X_train_lbp = lbph(X_train_reshaped)
    X_test_lbp = lbph(X_test_reshaped)
    
    # Define methods to evaluate
    methods = {
        'PCA': (X_train_pca, X_test_pca),
        'LDA': (X_train_lda, X_test_lda),
        'LBP': (X_train_lbp, X_test_lbp)
    }
    
    # Evaluate each method
    for method_name, (X_train_method, X_test_method) in methods.items():
        logger.info(f"Evaluating {method_name}...")
        
        # Train k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(X_train_method, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test_method)
        y_pred_proba = knn.predict_proba(X_test_method)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"{method_name} accuracy: {accuracy:.4f}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate TPR and FPR for each class
        tpr = {}
        fpr = {}
        roc_auc = {}
        
        # Calculate metrics for each class (one-vs-rest)
        for i in range(n_classes):
            # Create binary labels
            y_true_binary = (y_test == i).astype(int)
            y_score = y_pred_proba[:, i] if i < y_pred_proba.shape[1] else np.zeros(len(y_test))
            
            # Calculate ROC curve
            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate micro-average ROC curve
        y_test_binary = np.eye(n_classes)[y_test.astype(int)]
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binary.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Store results
        results[method_name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'true_labels': y_test
        }
    
    return results

def visualize_classification_metrics(results, person_names):
    methods = list(results.keys())
    n_classes = len(person_names)
    
    for method in methods:
        logger.info(f"Visualizing classification metrics for {method}...")
        
        # Get predictions and true labels
        y_pred = results[method]['predictions']
        y_test = results[method]['true_labels']
        
        # Initialize matrices
        tp_matrix = np.zeros(n_classes)
        fp_matrix = np.zeros(n_classes)
        tn_matrix = np.zeros(n_classes)
        fn_matrix = np.zeros(n_classes)
        
        # Calculate metrics for each class
        for i in range(n_classes):
            # True values for current class (binary)
            true_binary = (y_test == i).astype(int)
            # Predicted values for current class (binary)
            pred_binary = (y_pred == i).astype(int)
            
            # Calculate TP, FP, TN, FN
            tp_matrix[i] = np.sum((true_binary == 1) & (pred_binary == 1))
            fp_matrix[i] = np.sum((true_binary == 0) & (pred_binary == 1))
            tn_matrix[i] = np.sum((true_binary == 0) & (pred_binary == 0))
            fn_matrix[i] = np.sum((true_binary == 1) & (pred_binary == 0))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Classification Metrics for {method}', fontsize=16)
        
        # Plot TP
        axes[0, 0].bar(range(n_classes), tp_matrix, color='green')
        axes[0, 0].set_title('True Positives')
        axes[0, 0].set_xticks(range(n_classes))
        axes[0, 0].set_xticklabels(person_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot FP
        axes[0, 1].bar(range(n_classes), fp_matrix, color='red')
        axes[0, 1].set_title('False Positives')
        axes[0, 1].set_xticks(range(n_classes))
        axes[0, 1].set_xticklabels(person_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot TN
        axes[1, 0].bar(range(n_classes), tn_matrix, color='blue')
        axes[1, 0].set_title('True Negatives')
        axes[1, 0].set_xticks(range(n_classes))
        axes[1, 0].set_xticklabels(person_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot FN
        axes[1, 1].bar(range(n_classes), fn_matrix, color='orange')
        axes[1, 1].set_title('False Negatives')
        axes[1, 1].set_xticks(range(n_classes))
        axes[1, 1].set_xticklabels(person_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'classification_metrics_{method}.png')
        plt.show()
        
        # Calculate derived metrics
        precision = np.divide(tp_matrix, tp_matrix + fp_matrix, 
                             out=np.zeros_like(tp_matrix), where=(tp_matrix + fp_matrix) != 0)
        recall = np.divide(tp_matrix, tp_matrix + fn_matrix,
                          out=np.zeros_like(tp_matrix), where=(tp_matrix + fn_matrix) != 0)
        f1_scores = np.divide(2 * precision * recall, precision + recall,
                            out=np.zeros_like(precision), where=(precision + recall) != 0)
        
        # Plot precision, recall and F1 score
        plt.figure(figsize=(12, 6))
        x = np.arange(len(person_names))
        width = 0.25
        
        plt.bar(x - width, precision, width=width, label='Precision', color='skyblue')
        plt.bar(x, recall, width=width, label='Recall', color='lightgreen')
        plt.bar(x + width, f1_scores, width=width, label='F1 Score', color='salmon')
        
        plt.xlabel('Person')
        plt.ylabel('Score')
        plt.title(f'Precision, Recall and F1 Score for {method}')
        plt.xticks(x, person_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'precision_recall_f1_{method}.png')
        plt.show()

def visualize_results(results, person_names):
    methods = list(results.keys())
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    accuracies = [results[method]['accuracy'] for method in methods]
    plt.bar(methods, accuracies, color=['blue', 'green', 'red'])
    plt.title('Accuracy Comparison Across Methods')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()
    
    # Plot confusion matrices
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    
    for i, method in enumerate(methods):
        ax = axes[i] if n_methods > 1 else axes
        cm = results[method]['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix - {method}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels
        tick_marks = np.arange(len(person_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(person_names, rotation=45, ha='right')
        ax.set_yticklabels(person_names)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, format(cm[i, j], 'd'), 
                    ha="center", va="center", 
                    color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    for method in methods:
        # Plot micro-average ROC curve for each method
        plt.plot(results[method]['fpr']["micro"], results[method]['tpr']["micro"],
                label=f'{method} (AUC = {results[method]["roc_auc"]["micro"]:.3f})',
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison (Micro-average)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.show()
    
    # Plot additional metrics (TPR/FPR per class)
    for method in methods:
        plt.figure(figsize=(12, 8))
        
        # Get class-specific metrics
        fpr_dict = results[method]['fpr']
        tpr_dict = results[method]['tpr']
        roc_auc_dict = results[method]['roc_auc']
        
        # Plot ROC curve for each class
        for i, person in enumerate(person_names):
            if i in fpr_dict and i in tpr_dict and i in roc_auc_dict:
                plt.plot(fpr_dict[i], tpr_dict[i],
                        label=f'{person} (AUC = {roc_auc_dict[i]:.3f})')
        
        # Plot micro-average ROC curve
        plt.plot(fpr_dict["micro"], tpr_dict["micro"],
                label=f'Micro-average (AUC = {roc_auc_dict["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=3)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {method}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'roc_curves_{method}.png')
        plt.show()

        visualize_classification_metrics(results, person_names)

def main():
    dataset_dir = './dataset'
    scan_dir = './scans'
    
    logger.info("Starting biometric recognition simulation")
    
    # Check if directories exist
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return
    
    if not os.path.exists(scan_dir):
        logger.warning(f"Scans directory not found: {scan_dir}")
        logger.info("Will use part of training data for testing")
    
    # Load dataset
    X_train, X_test, y_train, y_test, person_names = load_dataset(dataset_dir, scan_dir)
    
    if X_train is None or len(X_train) == 0:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    logger.info(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    logger.info(f"People in dataset: {', '.join(person_names)}")
    
    # Evaluate all methods
    results = evaluate_classifier(X_train, X_test, y_train, y_test, person_names)
    
    # Visualize results
    visualize_results(results, person_names)
    
    logger.info("Simulation completed successfully!")

if __name__ == "__main__":
    main()