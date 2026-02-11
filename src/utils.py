import torch
from transformers import BertTokenizer, BertModel
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classifier
clf = joblib.load('models/fake_news_clf.pkl')

# Load BERT
tokenizer = BertTokenizer.from_pretrained('models/bert_tokenizer')
bert_model = BertModel.from_pretrained('models/bert_model').to(device)
bert_model.eval()

SATIRE_KEYWORDS = [
    "absolutely no one",
    "groundbreaking study",
    "researchers confirm",
    "probably",
    "emotionally prepared",
    "main character",
    "extremely serious",
    "mildly powerful",
    "we don't know why",
    "no one asked for"
]


def satire_heuristic(text):
    return any(word in text.lower() for word in SATIRE_KEYWORDS)


def predict_fake_news(text):

    # 🚨 SATIRE CHECK FIRST
    if satire_heuristic(text):
        return "Satire", 0.99

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)

    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    pred = clf.predict(cls_emb)
    prob = clf.predict_proba(cls_emb).max()

    return "Real" if pred[0] == 1 else "Fake", prob
