import os
import time
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Constants
DATASET_PATH = "dataset"
SCANS_PATH = "scans"
OUTPUT_FILE = "output.txt"
N_COMPONENTS = 50  # Number of principal components

def load_dataset():
    """Load images from the dataset directory"""
    images = []
    labels = []
    person_names = []
    
    for person_id, person_folder in enumerate(sorted(os.listdir(DATASET_PATH))):
        person_path = os.path.join(DATASET_PATH, person_folder)
        if not os.path.isdir(person_path):
            continue
            
        person_names.append(person_folder)
        
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (100, 100))
                images.append(img.flatten())
                labels.append(person_id)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), person_names

def prepare_data():
    """Prepare training and testing data"""
    X, y, person_names = load_dataset()
    
    # If scans directory exists, use it for testing
    if os.path.exists(SCANS_PATH) and os.listdir(SCANS_PATH):
        X_train, y_train = X, y
        X_test, y_test = load_scans(person_names)
    else:
        # Otherwise, split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        y_test = [person_names[label] for label in y_test]
    
    return X_train, y_train, X_test, y_test, person_names

def load_scans(person_names):
    """Load test images from the scans directory"""
    scans = []
    true_labels = []
    
    for img_file in os.listdir(SCANS_PATH):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(SCANS_PATH, img_file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Extract person name from filename
            file_basename = os.path.splitext(img_file)[0]
            extracted_name = file_basename.split('_')[0]  # Get first name
            
            # Match with the dataset name format
            matched_name = None
            for name in person_names:
                if name.lower() == extracted_name.lower() or name.startswith(extracted_name):
                    matched_name = name
                    break
            
            if matched_name is None:
                print(f"Warning: Could not match scan {img_file} to any person in dataset")
                continue
                
            img = cv2.resize(img, (100, 100))
            scans.append(img.flatten())
            true_labels.append(matched_name)
        except Exception as e:
            print(f"Error loading scan {img_path}: {e}")
    
    return np.array(scans), true_labels

def pca_svd(X, n_components):
    """Perform PCA using SVD"""
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    return components

def pca_krasulina(X, n_components, max_iter=1000, learning_rate=0.01):
    """Implement Krasulina's method for PCA"""
    n_samples, n_features = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    components = np.random.rand(n_components, n_features)
    components, _ = np.linalg.qr(components.T)
    components = components.T
    
    for _ in range(max_iter):
        idx = np.random.randint(0, n_samples)
        x = X_centered[idx]
        
        for j in range(n_components):
            projection = np.dot(x, components[j])
            components[j] += learning_rate * (projection * x - projection**2 * components[j])
            
            if j > 0:
                for k in range(j):
                    components[j] -= np.dot(components[j], components[k]) * components[k]
            
            norm = np.linalg.norm(components[j])
            if norm > 0:
                components[j] /= norm
    
    return components

def pca_oja(X, n_components, max_iter=1000, learning_rate=0.01):
    """Implement Oja's method for PCA"""
    n_samples, n_features = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    components = np.zeros((n_components, n_features))
    
    for j in range(n_components):
        w = np.random.rand(n_features)
        w /= np.linalg.norm(w)
        
        # Deflate the data for subsequent components
        if j > 0:
            X_proj = np.dot(X_centered, components[:j].T)
            X_approx = np.dot(X_proj, components[:j])
            X_residual = X_centered - X_approx
        else:
            X_residual = X_centered
            
        for _ in range(max_iter):
            idx = np.random.randint(0, n_samples)
            x = X_residual[idx]
            
            y = np.dot(w, x)
            w += learning_rate * y * (x - y * w)
            w /= np.linalg.norm(w)
        
        components[j] = w
    
    return components

def pca_sanger(X, n_components, max_iter=1000, learning_rate=0.01):
    """Implement Sanger's method (GHA) for PCA"""
    if len(X) == 0:
        raise ValueError("Empty training set")
        
    n_samples, n_features = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    components = np.random.rand(n_components, n_features)
    components /= np.linalg.norm(components, axis=1, keepdims=True)
    
    for _ in range(max_iter):
        idx = np.random.randint(0, n_samples)
        x = X_centered[idx]
        
        y = np.dot(components, x)
        
        for j in range(n_components):
            # Calculate lower triangular y*w terms
            lower_sum = np.zeros(n_features)
            for k in range(j+1):
                lower_sum += y[k] * components[k]
            
            # Update rule
            components[j] += learning_rate * y[j] * (x - lower_sum)
            
            # Normalize
            components[j] /= np.linalg.norm(components[j])
    
    return components

def pca_sanger_optimized(X, n_components, max_iter=500, initial_lr=0.01, 
                         lr_decay=0.01, batch_size=16):
    """Time-constrained implementation of Sanger's method with safety measures"""
    n_samples, n_features = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    # Scale data to prevent numerical issues
    scale = np.max(np.abs(X_centered)) + 1e-10
    X_centered = X_centered / scale
    
    # Use SVD for initialization - this gives better starting point
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    
    # Set a maximum time for the algorithm (20 seconds)
    max_time = 20  # seconds
    start_time = time.time()
    
    iteration = 0
    while iteration < max_iter:
        # Check timeout
        if time.time() - start_time > max_time:
            print(f"Sanger optimization timed out after {iteration} iterations")
            break
            
        # Adaptive learning rate
        learning_rate = initial_lr / (1 + lr_decay * iteration)
        
        # Process a batch of random samples
        batch_indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
        X_batch = X_centered[batch_indices]
        
        # Get projections
        Y = np.dot(X_batch, components.T)  # Shape: (batch_size, n_components)
        
        # Update each component
        for i in range(n_components):
            y_i = Y[:, i].reshape(-1, 1)  # Shape: (batch_size, 1)
            
            if i > 0:
                # Safe implementation of lateral inhibition using proper broadcasting
                inhibition_term = np.zeros_like(X_batch)  # Shape: (batch_size, n_features)
                for j in range(i):
                    y_j = Y[:, j].reshape(-1, 1)  # Shape: (batch_size, 1)
                    inhibition_term += y_j * components[j].reshape(1, -1)  # Broadcasting to (batch_size, n_features)
                
                # Calculate updates with proper shapes
                update1 = np.mean(y_i * X_batch, axis=0)  # Shape: (n_features,)
                update2 = np.mean(y_i * inhibition_term, axis=0)  # Shape: (n_features,)
                delta = learning_rate * (update1 - update2)
                delta = np.clip(delta, -0.01, 0.01)
            else:
                # For the first component, no inhibition needed
                update = np.mean(y_i * X_batch, axis=0)  # Shape: (n_features,)
                delta = learning_rate * update
                delta = np.clip(delta, -0.01, 0.01)
            
            components[i] += delta
        
        # Orthonormalize every iteration for stability
        components, _ = np.linalg.qr(components.T)
        components = components.T
        
        iteration += 1
    
    return components

def authenticate(test_image, train_images, train_labels, components, person_names):
    """Authenticate a test image against the training set in PCA space"""
    train_mean = np.mean(train_images, axis=0)
    
    # Project training data to PCA space
    train_centered = train_images - train_mean
    train_projection = np.dot(train_centered, components.T)
    
    # Project test image to PCA space
    test_centered = test_image - train_mean
    test_projection = np.dot(test_centered, components.T)
    
    # Find nearest neighbor in PCA space
    distances = np.linalg.norm(train_projection - test_projection, axis=1)
    nearest_idx = np.argmin(distances)
    
    return person_names[train_labels[nearest_idx]]

def evaluate_method(name, train_fn, X_train, y_train, X_test, y_test, person_names):
    """Evaluate a PCA method and return performance metrics"""
    try:
        # Train the PCA model
        start_time = time.time()
        components = train_fn(X_train, N_COMPONENTS)
        train_time = time.time() - start_time
        
        # Authenticate test samples
        start_time = time.time()
        y_pred = []
        for i in range(len(X_test)):
            pred = authenticate(X_test[i], X_train, y_train, components, person_names)
            y_pred.append(pred)
            print(f"Test {i}: Expected '{y_test[i]}', Predicted '{pred}'")
        auth_time = time.time() - start_time
        
        # Calculate accuracy
        correct = sum(1 for i in range(len(y_test)) if y_pred[i] == y_test[i])
        accuracy = correct / len(y_test) if len(y_test) > 0 else 0
        
        return f"Method: {name}\nTraining Time: {train_time:.4f} seconds\nAuthentication Time: {auth_time:.4f} seconds\nAccuracy: {accuracy:.4f}\n"
    except Exception as e:
        return f"Method: {name}\nTraining Time: Error - {str(e)}\nAuthentication Time: N/A\nAccuracy: N/A\n"

def create_pca_visualization():
    """Create a simple standalone visualization with hardcoded values if needed"""
    try:
        # Read the results from the output file - more reliable
        with open(OUTPUT_FILE, 'r') as f:
            content = f.read()
        
        results = content.split('\n\n')
        methods = []
        train_times = []
        auth_times = []
        accuracies = []
        
        for result in results:
            if not result.strip():
                continue
                
            lines = result.strip().split('\n')
            if len(lines) < 4:
                continue
                
            method = lines[0].split(': ')[1]
            
            # Skip methods with errors
            if 'Error' in lines[1]:
                continue
                
            train_time = float(lines[1].split(': ')[1].split(' ')[0])
            auth_time = float(lines[2].split(': ')[1].split(' ')[0])
            accuracy = float(lines[3].split(': ')[1])
            
            methods.append(method)
            train_times.append(train_time)
            auth_times.append(auth_time)
            accuracies.append(accuracy)
            
        if not methods:
            raise ValueError("No valid results found in output.txt")
            
        # Create simple plots with clear axes
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Training Time
        x = np.arange(len(methods))
        axes[0, 0].bar(x, train_times, color='skyblue')
        axes[0, 0].set_title("Training Time (lower is better)")
        axes[0, 0].set_ylabel("Seconds")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods, rotation=20)
        
        # 2. Authentication Time
        axes[0, 1].bar(x, auth_times, color='lightgreen')
        axes[0, 1].set_title("Authentication Time")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(methods, rotation=20)
        
        # 3. Accuracy
        axes[1, 0].bar(x, accuracies, color='salmon')
        axes[1, 0].set_title("Accuracy (higher is better)")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(methods, rotation=20)
        
        # 4. Scatter plot
        axes[1, 1].scatter(train_times, accuracies, s=200, alpha=0.7)
        for i, method in enumerate(methods):
            axes[1, 1].annotate(method, (train_times[i], accuracies[i]))
        axes[1, 1].set_title("Accuracy vs. Training Time")
        axes[1, 1].set_xlabel("Training Time (seconds)")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("pca_comparison.png")
        print("Visualization saved as 'pca_comparison.png'")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Load dataset and create train/test split
    X_train, y_train, X_test, y_test, person_names = prepare_data()
    
    # Evaluate each PCA method sequentially
    print("\nRUNNING PCA ALGORITHMS")
    print("=====================")
    results = []
    
    print("\nTesting SVD method:")
    results.append(evaluate_method("SVD", pca_svd, X_train, y_train, X_test, y_test, person_names))
    
    print("\nTesting Krasulina method:")
    results.append(evaluate_method("Krasulina", pca_krasulina, X_train, y_train, X_test, y_test, person_names))
    
    print("\nTesting Oja method:")
    results.append(evaluate_method("Oja", pca_oja, X_train, y_train, X_test, y_test, person_names))
    
    print("\nTesting standard Sanger method:")
    results.append(evaluate_method("Sanger", pca_sanger, X_train, y_train, X_test, y_test, person_names))
    
    print("\nTesting optimized Sanger method:")
    results.append(evaluate_method("Sanger-Opt", pca_sanger_optimized, X_train, y_train, X_test, y_test, person_names))
    
    # Write results to output file
    print("\nWriting results to output.txt")
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n\n'.join(results))
    
    # Create a separate standalone visualization
    print("\nGenerating visualization...")
    create_pca_visualization()

if __name__ == "__main__":
    main()