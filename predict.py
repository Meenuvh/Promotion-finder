import joblib

# Load saved model and vectorizer
clf = joblib.load("text_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_promotion(text):
    # Preprocess and predict
    vec = vectorizer.transform([text])
    prediction = clf.predict(vec)[0]
    return prediction

# Test it
if __name__ == "__main__":
    print("🧠 Promotion Classifier Ready.")
    while True:
        user_input = input("\nEnter text (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        label = predict_promotion(user_input)
        print(f"➡ Prediction: {label}")