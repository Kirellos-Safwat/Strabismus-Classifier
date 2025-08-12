import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, clone_model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple

# ---------------- Existing training utilities (unchanged helpers below) ---------------- #

def create_individual_datasets():
    """Create individual binary datasets for each strabismus type vs normal"""
    base_path = 'STRABISMUS'
    strabismus_types = ['ESOTROPIA', 'EXOTROPIA', 'HYPERTROPIA', 'HYPOTROPIA']
    
    datasets = {}
    
    for strab_type in strabismus_types:
        dataset_name = f'BINARY_{strab_type}'
        dataset_path = dataset_name
        
        # Create directories
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'NORMAL'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, strab_type), exist_ok=True)
        
        # Copy normal images
        normal_source = os.path.join(base_path, 'NORMAL')
        normal_dest = os.path.join(dataset_path, 'NORMAL')
        
        if os.path.exists(normal_source):
            for img in os.listdir(normal_source):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(os.path.join(normal_source, img), os.path.join(normal_dest, img))
        
        # Copy specific strabismus type images
        strab_source = os.path.join(base_path, strab_type)
        strab_dest = os.path.join(dataset_path, strab_type)
        
        if os.path.exists(strab_source):
            for img in os.listdir(strab_source):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(os.path.join(strab_source, img), os.path.join(strab_dest, img))
        
        datasets[strab_type] = dataset_path
        print(f"Created dataset for {strab_type}: {dataset_path}")
    
    return datasets


def build_individual_classifier(dataset_path, strab_type):
    """Build and train an individual binary classifier for normal vs specific strabismus type"""
    print(f"\nBuilding classifier for NORMAL vs {strab_type}")
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=16,  # Smaller batch size for individual models
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )
    
    # Build model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        f'best_{strab_type.lower()}_classifier.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print(f"Training {strab_type} classifier...")
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate model
    validation_generator.reset()
    predictions = model.predict(validation_generator)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = validation_generator.classes
    
    print(f"\nClassification Report for {strab_type}:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=['NORMAL', strab_type]))
    
    return model, history, predictions.flatten()


def extract_features_from_models(models, image_paths):
    """Extract features from all individual models for meta-classifier training"""
    print("\nExtracting features from individual models...")
    

    all_features = []
    all_labels = []
    
    for img_path in image_paths:
        # Load and preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions from all models
        model_predictions = []
        for model in models:
            pred = model.predict(img_array, verbose=0)[0][0]
            model_predictions.append(pred)
        
        all_features.append(model_predictions)
        
        # Determine label (normal vs abnormal)
        true_class = img_path.split(os.sep)[-2]
        if true_class == 'NORMAL':
            all_labels.append(0)  # Normal
        else:
            all_labels.append(1)  # Abnormal
    
    return np.array(all_features), np.array(all_labels)


def build_meta_classifier(features, labels):
    """Build meta-classifier using Random Forest"""
    print("\nBuilding meta-classifier...")
    
    # Split data for meta-classifier
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train Random Forest meta-classifier
    meta_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    meta_classifier.fit(X_train, y_train)
    
    # Evaluate meta-classifier
    train_score = meta_classifier.score(X_train, y_train)
    test_score = meta_classifier.score(X_test, y_test)
    
    print(f"Meta-classifier Training Accuracy: {train_score:.4f}")
    print(f"Meta-classifier Test Accuracy: {test_score:.4f}")
    
    # Feature importance
    feature_names = ['ESOTROPIA_Score', 'EXOTROPIA_Score', 'HYPERTROPIA_Score', 'HYPOTROPIA_Score']
    importance = meta_classifier.feature_importances_
    
    print("\nFeature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.4f}")
    
    return meta_classifier


def create_ensemble_model(individual_models, meta_classifier):
    """Create a combined ensemble model by cloning submodels with unique layer names"""
    print("\nCreating ensemble model...")

    # Shared input
    input_layer = Input(shape=(224, 224, 3))

    individual_predictions = []
    for i, model in enumerate(individual_models):
        prefix = f"mdl{i+1}"

        def prefix_clone_fn(layer):
            try:
                config = layer.get_config()
                # Ensure unique layer names by prefixing
                if 'name' in config and isinstance(config['name'], str):
                    config['name'] = f"{prefix}_{config['name']}"
                new_layer = layer.__class__.from_config(config)
                return new_layer
            except Exception:
                # Fallback: return layer class default if config not available
                return layer.__class__.from_config(layer.get_config())

        # Clone with prefixed names to avoid collisions (e.g., multiple 'vgg16')
        cloned = clone_model(model, clone_function=prefix_clone_fn)
        cloned.set_weights(model.get_weights())
        cloned.trainable = False

        # Apply cloned model to shared input
        pred = cloned(input_layer)
        individual_predictions.append(pred)

    concatenated = Concatenate()(individual_predictions)

    meta_output = Dense(64, activation='relu')(concatenated)
    meta_output = Dropout(0.3)(meta_output)
    meta_output = Dense(1, activation='sigmoid')(meta_output)

    ensemble_model = Model(inputs=input_layer, outputs=meta_output)
    ensemble_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    print("Ensemble model created.")
    return ensemble_model


def _binarize_true_from_generator(test_generator) -> np.ndarray:
    true_classes_raw = test_generator.classes
    class_indices = getattr(test_generator, 'class_indices', None)
    normal_index = None
    if isinstance(class_indices, dict) and class_indices:
        if 'NORMAL' in class_indices:
            normal_index = class_indices['NORMAL']
        else:
            for k, v in class_indices.items():
                if str(k).lower() == 'normal':
                    normal_index = v
                    break
    if normal_index is None:
        # Fallback: assume label 0 is NORMAL when 0 exists
        normal_index = 0
    return (true_classes_raw != normal_index).astype(int)


def evaluate_ensemble(ensemble_model, test_generator) -> Tuple[np.ndarray, float]:
    """Evaluate the ensemble model; returns (probabilities, accuracy)."""
    print("\nEvaluating ensemble model...")
    
    test_generator.reset()
    predictions = ensemble_model.predict(test_generator)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # Binary ground truth
    true_classes = _binarize_true_from_generator(test_generator)
    
    print("\nEnsemble Classification Report:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=['NORMAL', 'ABNORMAL']))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'ABNORMAL'],
                yticklabels=['NORMAL', 'ABNORMAL'])
    plt.title('Ensemble Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    acc = float(np.mean(predicted_classes == true_classes))
    return predictions.flatten(), acc


def evaluate_meta_ensemble(individual_models, meta_classifier, test_generator) -> Tuple[np.ndarray, float]:
    """Evaluate meta-ensemble using individual model sigmoid scores and a meta-classifier."""
    if meta_classifier is None:
        raise ValueError("meta_classifier is None; cannot run meta-ensemble evaluation")

    print("\nEvaluating meta-ensemble (RandomForest) on dataset...")

    n_samples = len(test_generator.filenames)

    # Collect scores from each individual model by predicting over the whole generator
    model_scores = []
    for idx, mdl in enumerate(individual_models):
        test_generator.reset()
        # Predict probabilities/scores; shape (n, 1) or (n,)
        scores = mdl.predict(test_generator, verbose=0)
        scores = scores.reshape(-1)[:n_samples]
        model_scores.append(scores)

    # Shape (n_samples, num_models)
    X = np.vstack(model_scores).T

    # Meta predictions
    meta_preds = meta_classifier.predict(X).astype(int)
    true_classes = _binarize_true_from_generator(test_generator)

    print("\nMeta-Ensemble Classification Report:")
    print(classification_report(true_classes, meta_preds, target_names=['NORMAL', 'ABNORMAL']))

    cm = confusion_matrix(true_classes, meta_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['NORMAL', 'ABNORMAL'],
                yticklabels=['NORMAL', 'ABNORMAL'])
    plt.title('Meta-Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('meta_ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    acc = float(np.mean(meta_preds == true_classes))
    return meta_preds, acc


def visualize_ensemble_performance(individual_accuracies, ensemble_accuracy):
    """Visualize performance comparison"""
    models = ['ESOTROPIA', 'EXOTROPIA', 'HYPERTROPIA', 'HYPOTROPIA', 'Ensemble']
    accuracies = individual_accuracies + [ensemble_accuracy]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title('Individual vs Ensemble Model Performance')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('ensemble_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------- High-level entrypoints --------------------------- #

def train_all():
    print("Ensemble Learning for Strabismus Classification")
    print("=" * 60)
    
    # Step 1: Create individual datasets
    print("Step 1: Creating individual datasets...")
    datasets = create_individual_datasets()
    
    # Step 2: Train individual classifiers
    print("\nStep 2: Training individual classifiers...")
    individual_models = []
    individual_accuracies = []
    strabismus_types = ['ESOTROPIA', 'EXOTROPIA', 'HYPERTROPIA', 'HYPOTROPIA']
    
    for strab_type in strabismus_types:
        model, history, _ = build_individual_classifier(datasets[strab_type], strab_type)
        individual_models.append(model)
        individual_accuracies.append(max(history.history['val_accuracy']))
    
    # Step 3: Prepare data for meta-classifier
    print("\nStep 3: Preparing data for meta-classifier...")
    
    # Collect all image paths
    all_image_paths = []
    base_path = 'STRABISMUS'
    
    for class_name in ['NORMAL'] + strabismus_types:
        class_path = os.path.join(base_path, class_name)
        if os.path.exists(class_path):
            for img in os.listdir(class_path):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    all_image_paths.append(os.path.join(class_path, img))
    
    # Extract features
    features, labels = extract_features_from_models(individual_models, all_image_paths)
    
    # Step 4: Build meta-classifier
    print("\nStep 4: Building meta-classifier...")
    meta_classifier = build_meta_classifier(features, labels)
    
    # Save meta-classifier
    joblib.dump(meta_classifier, 'meta_classifier.pkl')
    print("Meta-classifier saved as 'meta_classifier.pkl'")
    
    # Step 5: Create ensemble model
    print("\nStep 5: Creating ensemble model...")
    ensemble_model = create_ensemble_model(individual_models, meta_classifier)
    
    # Step 6: Evaluate ensemble
    print("\nStep 6: Evaluating ensemble model...")
    
    # Create test generator for evaluation
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'STRABISMUS',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    _, ensemble_accuracy = evaluate_ensemble(ensemble_model, test_generator)
    
    # Step 7: Visualize performance
    print("\nStep 7: Visualizing performance...")
    visualize_ensemble_performance(individual_accuracies, ensemble_accuracy)
    
    print("\nEnsemble Learning Complete!")
    print(f"Individual Model Accuracies: {[f'{acc:.3f}' for acc in individual_accuracies]}")
    print(f"Ensemble Model Accuracy: {ensemble_accuracy:.3f}")
    
    # Save ensemble model
    ensemble_model.save('ensemble_model.h5')
    print("Ensemble model saved as 'ensemble_model.h5'")


def assemble_from_existing():
    print("Start from Step 5: Build Ensemble from Existing Models")
    print("=" * 60)

    model_files = {
        'ESOTROPIA': 'best_esotropia_classifier.h5',
        'EXOTROPIA': 'best_exotropia_classifier.h5',
        'HYPERTROPIA': 'best_hypertropia_classifier.h5',
        'HYPOTROPIA': 'best_hypotropia_classifier.h5',
    }

    # Load individual models
    individual_models = []
    missing = []
    for key, path in model_files.items():
        if not os.path.exists(path):
            missing.append((key, path))
        else:
            individual_models.append(load_model(path))
            print(f"Loaded model: {key} from {path}")
    if missing:
        print("\nMissing required models:")
        for key, path in missing:
            print(f"- {key}: expected at {path}")
        print("\nPlace missing models and rerun.")
        return

    # Load meta-classifier (optional)
    meta_classifier = None
    if os.path.exists('meta_classifier.pkl'):
        try:
            meta_classifier = joblib.load('meta_classifier.pkl')
            print("Loaded meta-classifier: meta_classifier.pkl")
        except Exception as e:
            print(f"Warning: Failed to load meta-classifier: {e}")

    # Build ensemble (Keras head is untrained baseline)
    print("\nCreating ensemble model...")
    ensemble_model = create_ensemble_model(individual_models, meta_classifier)
    print("Ensemble model built successfully.")
    ensemble_model.save('ensemble_model.h5')
    print("Saved ensemble model to 'ensemble_model.h5'")

    # Optional evaluation
    base = 'STRABISMUS'
    if not os.path.isdir(base):
        print("Dataset directory 'STRABISMUS' not found. Skipping evaluation.")
        return
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        base, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
    )

    print("\nEvaluating ensemble on dataset (Keras head; untrained baseline)...")
    _, keras_acc = evaluate_ensemble(ensemble_model, test_gen)
    print(f"\nKeras-head ensemble accuracy (untrained head): {keras_acc:.3f}")

    if meta_classifier is not None:
        print("\nEvaluating meta-ensemble (RandomForest) using individual model scores...")
        _, meta_acc = evaluate_meta_ensemble(individual_models, meta_classifier, test_gen)
        print(f"\nMeta-ensemble accuracy: {meta_acc:.3f}")
    else:
        print("\nMeta-classifier not available; skipped meta-ensemble evaluation.")


# --------------------------- CLI dispatcher --------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strabismus ensemble pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Train all models and meta-classifier from scratch")
    sub.add_parser("assemble", help="Build/evaluate ensemble from existing models and meta-classifier")

    args = parser.parse_args()
    if args.cmd == "train":
        train_all()
    elif args.cmd == "assemble":
        assemble_from_existing()