import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Spam Detector API")

# Saytdan (Frontend) so'rovlar qabul qilish uchun CORS ruxsati
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")

with open("vocab_pro.pkl", "rb") as f:
    vocab = pickle.load(f)

class SpamBiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, n_layers=2, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, return_attention=False):
        mask = (x != 0)
        embedded = self.embed_drop(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        
        attn_scores = self.attention_weights(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_probs = F.softmax(attn_scores, dim=1)
        attn_context = torch.bmm(attn_probs.unsqueeze(1), lstm_out).squeeze(1)
        
        lstm_masked = lstm_out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pool, _ = torch.max(lstm_masked, dim=1)
        
        combined = self.layer_norm(torch.cat([attn_context, max_pool], dim=1))
        out = F.gelu(self.fc1(self.dropout(combined)))
        logit = self.fc2(self.dropout(out)).squeeze(1)
        
        if return_attention:
            return logit, attn_probs
        return logit

model = SpamBiLSTMAttention(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load("best_spam_bilstm_pro.pt", map_location=device))
model.eval()

def smart_clean_uzbek(text):
    text = str(text).replace("Ê»", "'").replace("Ê¼", "'").replace("â€™", "'")
    text = text.replace("o'", "oʻ").replace("g'", "gʻ").replace("o‘", "oʻ").replace("g‘", "gʻ").lower()
    text = re.sub(r't\.me/[a-z0-9_]+bot\b', ' <TG_BOT> ', text)
    text = re.sub(r'https?://\S+|www\.\S+|t\.me/\S+', ' <WEB_URL> ', text)
    text = re.sub(r'\b([01]?\d|2[0-3]):[0-5]\d\b', ' <TIME> ', text)
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', ' <CARD> ', text)
    text = re.sub(r'\+?\d[\d\s-]{7,}\d', ' <PHONE> ', text)
    text = re.sub(r'\b\d+[\d\s]*\s?(rubl|dollar|\$|som|soʻm|sum|yevro|million)\b', ' <MONEY> ', text)
    text = re.sub(r'[^a-zʻoʻgʻа-яёA-Z\s\<\>_]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

MAX_LEN = 60

class MessageRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "online", "model": "Bi-LSTM Attention"}

@app.post("/predict")
def predict_threat(data: MessageRequest):
    raw = data.message
    cleaned = smart_clean_uzbek(raw)
    tokens = cleaned.split()[:MAX_LEN]
    
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
        
    tensor_in = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logit, attn_weights = model(tensor_in, return_attention=True)
        prob = torch.sigmoid(logit).item()
        
    weights = attn_weights[0][:len(tokens)].cpu().numpy()
    triggers = []
    if len(weights) > 0 and weights.sum() > 0:
        weights = weights / weights.sum()
        top_indices = weights.argsort()[::-1][:4]
        triggers = [f"{tokens[i]} ({weights[i]*100:.1f}%)" for i in top_indices if i < len(tokens)]
        
    iocs = []
    if re.search(r'https?://\S+|www\.\S+|t\.me/\S+', raw):
        iocs.append({"icon": "link", "text": "Shubhali havola aniqlandi", "color": "text-amber-400"})
    if re.search(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', raw):
        iocs.append({"icon": "credit-card", "text": "16 xonali karta raqami", "color": "text-red-400"})
    if re.search(r'sms|kod|parol', raw, re.I):
        iocs.append({"icon": "key", "text": "SMS tasdiqlash kodi talabi", "color": "text-red-400"})
    if re.search(r't\.me/[a-z0-9_]+bot\b', raw):
        iocs.append({"icon": "bot", "text": "Telegram Bot yo'naltirmasi", "color": "text-cyan-400"})

    percent = round(prob * 100, 2)
    if prob >= 0.85:
        category = "🚨 YUQORI XAVF (FISHING / FIRIBGARLIK)"
        why = "Neyron tarmoq ushbu xabarda yuqori darajadagi manipulyatsiya, shubhali havola yoki shaxsiy ma'lumotlarni o'g'irlash alomatlarini aniqladi."
        advice_title = "ZUDLIK BILAN TO'XTATING:"
        advice_desc = "Hech qanday havolaga kirmang, karta ma'lumotlari yoki tasdiqlash kodini kiritmang!"
        risk_level = "danger"
    elif prob >= 0.50:
        category = "⚠️ O'RTACHA XAVF (PROMO / SPAM)"
        why = "Xabarda to'g'ridan-to'g'ri fishing alomatlari kam, ammo ommaviy reklama, botlar yoki kurs takliflari kabi tarqatma alomatlari mavjud."
        advice_title = "E'TIBOR TALAB QILINADI:"
        advice_desc = "Agar ushbu xizmatga o'zingiz yozilmagan bo'lsangiz, shunchaki e'tiborsiz qoldiring yoki botni bloklang."
        risk_level = "warn"
    else:
        category = "✅ XAVFSIZ (ODDIY XABAR)"
        why = "Matnda kiber-tahdid, firibgarlik yoki shubhali moliyaviy bosim aniqlanmadi. Xabar oddiy yoki rasmiy muloqot xarakteriga ega."
        advice_title = "XAVFSIZLIK TASDIQLANDI:"
        advice_desc = "Hech qanday kiberxavfsizlik tahdidi yo'q."
        risk_level = "safe"

    return {
        "prob": percent,
        "category": category,
        "why": why,
        "advice_title": advice_title,
        "advice_desc": advice_desc,
        "triggers": triggers,
        "iocs": iocs,
        "risk_level": risk_level
    }
