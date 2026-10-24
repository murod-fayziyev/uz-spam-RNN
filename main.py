import os
import re
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Uzbek Cyber Threat & Spam Detection API", version="2.0")

DEVICE = torch.device("cpu")
MODEL_PATH = "best_model.pth"
VOCAB_PATH = "vocab_pro.pkl"
MAX_LEN = 60

# --- 1. MODEL ARXITEKTURASI ---
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = torch.softmax(energy, dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights

class RobustCyberBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.35):
        super(RobustCyberBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = SelfAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)
        return out.squeeze(1), attn_weights

# --- 2. LUG'AT VA MODELNI YUKLASH ---
if not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("best_model.pth yoki vocab_pro.pkl topilmadi!")

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

model = RobustCyberBiLSTM(vocab_size=len(vocab)).to(DEVICE)
# MUHIM: Render CPU'da ishlashi uchun map_location='cpu'
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 3. TOKENIZATOR VA GIBRID FILTR ---
def tokenize_cyber(text):
    text = str(text).lower()
    text = re.sub(r'\b(8600|9860|5614|6262)[\d\s\*]{8,22}', ' <card> ', text)
    text = re.sub(r'\b[\w\-]+\.apk\b', ' <apk> ', text)
    text = re.sub(r'(\+?998[\s\-]?[0-9]{2}[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2})', ' <phone> ', text)
    text = re.sub(r'https?://\S+|www\.\S+|\b\S+\.(?:top|xyz|link|site|biz|ru|com|uz|info|online)\b\/?\S*', ' <url> ', text)
    text = re.sub(r'\b\d+[\d\s.,]*\b', ' <num> ', text)
    return re.findall(r'<card>|<url>|<apk>|<phone>|<num>|[\w\']+', text)

def text_to_sequence(text, vocab, max_len=MAX_LEN):
    tokens = tokenize_cyber(text)
    seq = [vocab.get(token, vocab.get('<unk>', 1)) for token in tokens]
    if len(seq) < max_len:
        seq = seq + [vocab.get('<pad>', 0)] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

TRUSTED_DOMAINS = [
    'uzcard.uz', 'humocard.uz', 'payme.uz', 'click.uz', 'uzum.uz',
    'kapitalbank.uz', 'nbu.uz', 'cbu.uz', 'anorbank.uz', 'tbcbank.uz',
    'ipakyulibank.uz', 'agrobank.uz', 'xalqbanki.uz', 'hamkorbank.uz',
    'gov.uz', 'my.gov.uz', 'lex.uz'
]

class MessageRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_threat(req: MessageRequest):
    text = req.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Xabar bo'sh bo'lishi mumkin emas.")

    tokens = tokenize_cyber(text)
    seq = text_to_sequence(text, vocab)
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(seq_tensor)
        ml_prob = torch.sigmoid(logits / 1.8).item()

    lower_t = text.lower()

    # Domen tahlili
    raw_urls = re.findall(r'(?:https?:\/\/)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
    has_untrusted_url = any(
        not any(dom.lower().replace('www.', '') == td or dom.lower().replace('www.', '').endswith('.' + td) for td in TRUSTED_DOMAINS)
        for dom in raw_urls
    )

    # Hujum belgilari
    has_apk = '<apk>' in tokens or '.apk' in lower_t
    has_card = bool(re.search(r'\b(8600|9860|5614|6262)[\d\s\*]{8,22}', text))
    urgency = ['kasal', 'shifoxona', 'bloklandi', 'tashlab tur', 'avariya', 'operatsiya', 'qarz', 'yordam ber', 'iib']
    has_social_eng = (has_card or 'karta' in lower_t) and any(w in lower_t for w in urgency)
    lures = ['yutdingiz', 'yutuq', 'sovrin', 'depozit bonusi', 'pul yutib', 'bepul tarqatilmoqda', 'kredit tasdiqlandi']
    has_lure = any(w in lower_t for w in lures)

    is_offensive = has_apk or has_social_eng or (has_untrusted_url and (has_lure or 'kod' in lower_t or 'kirish' in lower_t))

    # Rasmiy / Logistika belgilari (False Positive oldini oluvchi)
    defensive_phrases = [
        "hech kimga aytmang", "hech kimga bermang", "hech kim bilan bo'lishmang",
        "agar so'ramagan bo'lsangiz", "e'tibor bermang", "kredit to'lovi",
        "to'lashingiz kerak", "amal qilish muddati", "hisobingiz to'ldirildi",
        "kuryerga topshirildi", "yetkaziladi", "oralig'ida yetkaziladi",
        "buyurtmangiz", "kuryer:", "punktga yetib keldi"
    ]
    is_defensive = any(p in lower_t for p in defensive_phrases)

    if is_offensive:
        final_prob = max(ml_prob, 0.985)
        decision = "ANIQ KIBER-TAHDID"
    elif is_defensive and not has_untrusted_url and not has_card:
        final_prob = min(ml_prob, 0.035)
        decision = "RASMIY XIZMAT / LOGISTIKA"
    else:
        final_prob = ml_prob
        decision = "NEYRON TARMOQ TAHLILI"

    is_spam = final_prob >= 0.5
    return {
        "text": text,
        "is_threat": is_spam,
        "threat_probability": round(final_prob * 100, 2),
        "status": "XAVFLI (SPAM / FISHING)" if is_spam else "XAVFSIZ (RASMIY / ODDIY)",
        "decision_basis": decision
    }

@app.get("/")
def root():
    return {"status": "running", "model_version": "v2_balanced"}
