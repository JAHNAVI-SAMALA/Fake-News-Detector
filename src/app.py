from flask import Flask, render_template, request
from utils import predict_fake_news
import wikipedia

app = Flask(__name__)

def enriched_predict(text):
    # Add context if input is very short
    if len(text.split()) < 10:
        try:
            summary = wikipedia.summary(text, sentences=2)
            text_to_predict = summary
        except:
            text_to_predict = text
    else:
        text_to_predict = text

    label, confidence = predict_fake_news(text_to_predict)
    # Return both Real and Fake probabilities
    fake_prob = confidence if label == "Fake" else 1-confidence
    real_prob = 1 - fake_prob
    return {"label": label, "confidence": f"{confidence:.2f}", "fake_prob": fake_prob, "real_prob": real_prob}

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["news_text"]
        result = enriched_predict(text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
