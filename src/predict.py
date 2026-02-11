from utils import predict_fake_news
import wikipedia

def enriched_prediction(text):
    # If short sentence, try adding context from Wikipedia
    if len(text.split()) < 10:
        try:
            # Get a short summary from Wikipedia
            wiki_summary = wikipedia.summary(text, sentences=2)
            text_to_predict = wiki_summary
        except:
            text_to_predict = text  # fallback if not found
    else:
        text_to_predict = text

    label, confidence = predict_fake_news(text_to_predict)
    return label, confidence

# Test short sentence
text = "Taj Mahal is in India"
label, confidence = enriched_prediction(text)
print(f"Prediction: {label} | Confidence: {confidence:.2f}")
