from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# Cargar dataset en inglés
dataset = load_dataset("dair-ai/emotion")

texts = dataset["train"]["text"]
label_ids = dataset["train"]["label"]

# Obtener nombres de las emociones
label_names = dataset["train"].features["label"].names
labels = [label_names[i] for i in label_ids]

# Traducir etiquetas al español
label_translation = {
    'joy': 'alegría',
    'sadness': 'tristeza',
    'anger': 'enojo',
    'fear': 'miedo',
    'love': 'amor',
    'surprise': 'sorpresa'
}
translated_labels = [label_translation[label] for label in labels]

# Codificar etiquetas traducidas
le = LabelEncoder()
y = le.fit_transform(translated_labels)

# Crear y entrenar modelo
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])
model.fit(texts, y)

# Guardar modelo y codificador
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Modelo entrenado y guardado con etiquetas traducidas al español")