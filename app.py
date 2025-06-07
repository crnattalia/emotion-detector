from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Contador de emociones global
emotion_counter = {
    'alegría': 0,
    'tristeza': 0,
    'enojo': 0,
    'miedo': 0,
    'amor': 0,
    'sorpresa': 0
}

# Emojis para cada emoción
emotion_emojis = {
    'alegría': '😄',
    'tristeza': '😢',
    'enojo': '😠',
    'miedo': '😨',
    'amor': '❤️',
    'sorpresa': '😮'
}

# Cargar modelo y codificador
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html", 
                           emotion_counter=emotion_counter,
                           emotion_emojis=emotion_emojis)

@app.route("/predict", methods=["POST"])
def predict():
    global emotion_counter
    
    text = request.form["text"].strip()
    
    # Validación de entrada
    if not text or not any(char.isalpha() for char in text):
        return render_template(
            "index.html",
            error_message="Por favor ingresa un texto válido con contenido real",
            emotion_counter=emotion_counter,
            emotion_emojis=emotion_emojis
        )
    
    # Predicción
    prediction = model.predict([text])[0]
    emotion = label_encoder.inverse_transform([prediction])[0]
    
    # Obtener probabilidades
    probabilities = model.predict_proba([text])[0]
    emotion_probs = {}
    for i, class_name in enumerate(label_encoder.classes_):
        emotion_probs[class_name] = round(probabilities[i] * 100, 2)
    
    # Actualizar contador
    emotion_counter[emotion] += 1
    
    return render_template(
        "index.html",
        prediction=emotion,
        text=text,
        emotion_counter=emotion_counter,
        emotion_emojis=emotion_emojis,
        emotion_probs=emotion_probs,
        sorted_emotions=sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
    )

if __name__ == "__main__":
    app.run(debug=True)