import os
import argparse
import glob
import numpy as np
from typing import Dict, List, Tuple
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

MODEL_FILES = {
    'ESOTROPIA': 'best_esotropia_classifier.h5',
    'EXOTROPIA': 'best_exotropia_classifier.h5',
    'HYPERTROPIA': 'best_hypertropia_classifier.h5',
    'HYPOTROPIA': 'best_hypotropia_classifier.h5',
}


def load_artifacts() -> Tuple[List, object]:
    models = []
    for key, path in MODEL_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model for {key}: {path}")
        models.append(load_model(path))
    meta = None
    if os.path.exists('meta_classifier.pkl'):
        try:
            meta = joblib.load('meta_classifier.pkl')
        except Exception:
            meta = None
    return models, meta


def preprocess_image(image_path: str) -> np.ndarray:
    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img)
    arr = arr.astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_scores(img: np.ndarray, models: List) -> List[float]:
    scores = []
    for mdl in models:
        s = mdl.predict(img, verbose=0).reshape(-1)[0]
        scores.append(float(s))
    return scores


def combine_with_meta(scores: List[float], meta) -> Tuple[int, float]:
    # Returns (pred_label, confidence_for_pred)
    x = np.array(scores, dtype=float).reshape(1, -1)
    if meta is not None:
        # Use probability of class 1 (ABNORMAL)
        if hasattr(meta, 'predict_proba'):
            proba = meta.predict_proba(x)[0][1]
            pred = int(proba >= 0.5)
            conf = float(proba if pred == 1 else 1.0 - proba)
            return pred, conf
        pred = int(meta.predict(x)[0])
        # Fallback confidence: distance from 0.5 using mean score
        m = float(np.mean(scores))
        conf = abs(m - 0.5) * 2.0
        return pred, conf
    # Fallback: simple average
    m = float(np.mean(scores))
    pred = int(m >= 0.5)
    conf = float(m if pred == 1 else 1.0 - m)
    return pred, conf


def predict_image(image_path: str) -> Dict:
    models, meta = load_artifacts()
    arr = preprocess_image(image_path)
    scores = predict_scores(arr, models)
    pred, conf = combine_with_meta(scores, meta)
    result = {
        'image': image_path,
        'scores': {
            'ESOTROPIA': scores[0],
            'EXOTROPIA': scores[1],
            'HYPERTROPIA': scores[2],
            'HYPOTROPIA': scores[3],
        },
        'prediction': 'ABNORMAL' if pred == 1 else 'NORMAL',
        'confidence': conf,
    }
    return result


def run_folder(folder: str) -> List[Dict]:
    exts = ('*.jpg', '*.jpeg', '*.png')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    outputs = []
    for f in files:
        try:
            outputs.append(predict_image(f))
        except Exception as e:
            outputs.append({'image': f, 'error': str(e)})
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Predict NORMAL/ABNORMAL using ensemble meta-classifier.')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--folder', type=str, help='Folder of images (jpg/jpeg/png)')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error('Provide --image PATH or --folder DIR')

    if args.image:
        res = predict_image(args.image)
        print(f"Image: {res['image']}")
        print(f"Scores -> ESO: {res['scores']['ESOTROPIA']:.3f}, EXO: {res['scores']['EXOTROPIA']:.3f}, "
              f"HYPER: {res['scores']['HYPERTROPIA']:.3f}, HYPO: {res['scores']['HYPOTROPIA']:.3f}")
        print(f"Prediction: {res['prediction']}  Confidence: {res['confidence']:.3f}")
        return

    if args.folder:
        results = run_folder(args.folder)
        for r in results:
            if 'error' in r:
                print(f"Image: {r['image']}  ERROR: {r['error']}")
            else:
                print(f"Image: {r['image']}  Pred: {r['prediction']}  Conf: {r['confidence']:.3f}")
        return


if __name__ == '__main__':
    main()
