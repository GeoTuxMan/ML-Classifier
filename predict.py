import sys
import joblib

def predict(features):
    model = joblib.load("model.pkl")
    pred = model.predict([features])
    target_names = ['setosa', 'versicolor', 'virginica']
    return target_names[pred[0]]

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py sepal_length sepal_width petal_length petal_width")
        sys.exit(1)
    features = list(map(float, sys.argv[1:]))
    prediction = predict(features)
    print(f"Predicted class: {prediction}")
