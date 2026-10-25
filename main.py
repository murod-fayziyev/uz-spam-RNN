from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import os
import re
import pickle
import hashlib
import httpx
from typing import Optional

# =====================================================================
# 1. KONFIGURATSIYA VA SUPABASE SOZLAMALARI
# =====================================================================
SUPABASE_URL = "https://hmmdrxeotfjscwudsyhb.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

app = FastAPI(
    title="Uzbek Cyber Threat & SOC Intelligence API",
    version="3.0",
    description="Bi-LSTM Attention Engine + Supabase Threat Vault Integratsiyasi"
)

# CORS sozlamalari (GitHub Pages ruxsati)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 2. MODEL ARXITEKTURASI (Bi-LSTM + Self-Attention)
# =====================================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim * 2)
        scores = self.attn(lstm_output).squeeze(2)  # (batch_size, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)  # (batch_size, seq_len, 1)
        context = torch.sum(lstm_output * weights, dim=1)  # (batch_size, hidden_dim * 2)
        return context

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_layers=1):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_out = self.attention(lstm_out)
        out = self.fc(attn_out)
        return torch.sigmoid(out)

# =====================================================================
# 3. MODEL VA LUG'ATNI XOTIRAGA YUKLASH
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
vocab = None
MAX_LEN = 100

def load_ai_engine():
    global model, vocab
    vocab_path = "vocab_pro.pkl"
    model_path = "best_spam_bilstm_pro.pt"
    if not os.path.exists(model_path):
        model_path = "best_model.pth"

    try:
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            print(" [OK] Lug'at (vocab) muvaffaqiyatli yuklandi.")

        if os.path.exists(model_path) and vocab:
            model = BiLSTMClassifier(vocab_size=len(vocab))
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(" [OK] Bi-LSTM neyron modeli yuklandi.")
    except Exception as e:
        print(f" [OGOHLANTIRISH] Model yuklashda xatolik: {e}")

load_ai_engine()

# =====================================================================
# 4. KIBER-RAZVEDKA, PII NIQOBLASH VA IoC AJRATISH
# =====================================================================

def mask_personal_data(text: str) -> str:
    """Karta va telefon kabi shaxsiy ma'lumotlarni qisman yopish"""
    # Karta raqamlarini niqoblash: 8600 1234 5678 9012 -> 8600 12** **** 9012
    masked = re.sub(
        r'\b(8600|9860|5614|6262)[\s\-]?(\d{2})\d{2}[\s\-]?\d{4}[\s\-]?(\d{4})\b',
        r'\1 \2** **** \3',
        text
    )
    # Telefon raqamlarini niqoblash: +998 90 123 45 67 -> +998 90 *** ** 67
    masked = re.sub(
        r'(\+?998[\s\-]?[0-9]{2})[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}([\s\-]?[0-9]{2})',
        r'\1 *** ** \2',
        masked
    )
    return masked

def extract_threat_iocs(text: str):
    """Matndan texnik IoC parametrlarini ajratib olish"""
    lower = text.lower()
    
    # 1. Karta turi
    card_type = None
    if re.search(r'\b8600', text):
        card_type = "Uzcard (8600)"
    elif re.search(r'\b9860', text):
        card_type = "Humo (9860)"
    elif re.search(r'\b(5614|6262)', text):
        card_type = "Xalqaro / Boshqa Karta"

    # 2. Domen va domen zonasi
    domain_info = None
    url_match = re.search(r'https?://([^/\s]+)|www\.([^/\s]+)|\b([a-zA-Z0-9\.\-]+\.(?:top|xyz|link|site|biz|ru|com|uz|info|online))\b', text, re.I)
    if url_match:
        extracted = url_match.group(1) or url_match.group(2) or url_match.group(3)
        domain_info = extracted.lower()

    # 3. APK borligi
    has_apk = bool(".apk" in lower)

    return card_type, domain_info, has_apk

def calculate_text_hash(text: str) -> str:
    """Deduplication uchun xabar matnidan SHA-256 hesh olish"""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

# =====================================================================
# 5. SUPABASE BILAN ISHLASH (ASYNCHRONOUS VAULT ENGINE)
# =====================================================================

async def save_or_increment_in_vault(clean_text: str, text_hash: str, prob: float, card_type: str, domain: str, has_apk: bool):
    """Xabarni Supabase bazasiga kiritish yoki takrorlanish sonini oshirish"""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            # 1. Hesh orqali bazada bor-yo'qligini tekshirish
            check_url = f"{SUPABASE_URL}/rest/v1/threat_vault?text_hash=eq.{text_hash}&select=id,occurrence_count"
            resp = await client.get(check_url, headers=SUPABASE_HEADERS)
            
            if resp.status_code == 200 and resp.json():
                # Xabar allaqachon bor: occurrence_count ni +1 ga oshiramiz
                row = resp.json()[0]
                row_id = row["id"]
                current_count = row.get("occurrence_count", 1) or 1
                
                patch_url = f"{SUPABASE_URL}/rest/v1/threat_vault?id=eq.{row_id}"
                await client.patch(patch_url, headers=SUPABASE_HEADERS, json={"occurrence_count": current_count + 1})
            else:
                # Yangi xabar: 5-savat (inference_raw_logs) qilib yozamiz
                insert_url = f"{SUPABASE_URL}/rest/v1/threat_vault"
                payload = {
                    "text_hash": text_hash,
                    "clean_text": clean_text,
                    "model_prob": prob,
                    "user_reaction": "none",
                    "basket_type": "inference_raw_logs",
                    "ioc_card": card_type,
                    "ioc_domain": domain,
                    "ioc_has_apk": has_apk,
                    "occurrence_count": 1
                }
                await client.post(insert_url, headers=SUPABASE_HEADERS, json=payload)
    except Exception as e:
        print(f" [SUPABASE XATOLIGI] Yozishda muammo: {e}")

# =====================================================================
# 6. API ENDPOINTLARI
# =====================================================================

class MessageRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text_hash: str
    user_reaction: str  # 'correct' yoki 'incorrect'

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Uzbek Cyber Threat & SOC Classifier v3.0",
        "vault_connected": True
    }

@app.post("/predict")
async def predict_message(req: MessageRequest):
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Bo'sh matn kiritildi.")

    # 1. SHA-256 Hesh va Shaxsiy ma'lumotlarni niqoblash
    text_hash = calculate_text_hash(raw_text)
    clean_text = mask_personal_data(raw_text)
    card_type, domain, has_apk = extract_threat_iocs(raw_text)

    # 2. Xavf ehtimolini baholash (Model + Gibrid Kiber-Filtr)
    prob = 15.0  # Standart xavfsiz boshlang'ich
    basis = "Statik Evristik Qoidalar"

    if model and vocab:
        words = re.findall(r'\b\w+\b', raw_text.lower())
        seq = [vocab.get(w, 1) for w in words][:MAX_LEN]
        if len(seq) < MAX_LEN:
            seq += [0] * (MAX_LEN - len(seq))
        
        tensor_in = torch.tensor([seq], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(tensor_in).item()
            prob = round(out * 100, 1)
            basis = "Bi-LSTM Neyron Modeli"
    else:
        # Model fayli bo'lmaganda zaxira filtr
        if has_apk or (card_type and "http" in raw_text.lower()):
            prob = 92.0
            basis = "Evristik Kiber-Qoidalar (Zararli IoC)"
        elif "ovoz" in raw_text.lower() and "http" in raw_text.lower():
            prob = 88.0
            basis = "Telegram Fishing Qoidasi"

    # 3. Supabase bazasiga orqa fonda xavfsiz saqlash
    # (Foydalanuvchini kutib o'tirmasligi uchun tezkor yoziladi)
    await save_or_increment_in_vault(clean_text, text_hash, prob, card_type, domain, has_apk)

    return {
        "threat_probability": prob,
        "is_threat": prob >= 50.0,
        "text_hash": text_hash,
        "decision_basis": basis,
        "detected_iocs": {
            "card": card_type,
            "domain": domain,
            "has_apk": has_apk
        }
    }

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Foydalanuvchi fikriga asosan xabarni 5 ta savatga saralash"""
    reaction = req.user_reaction.lower()
    if reaction not in ["correct", "incorrect"]:
        raise HTTPException(status_code=400, detail="Noto'g'ri reaksiya turi.")

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            # Hesh orqali xabarni qidiramiz
            url = f"{SUPABASE_URL}/rest/v1/threat_vault?text_hash=eq.{req.text_hash}&select=id,model_prob"
            resp = await client.get(url, headers=SUPABASE_HEADERS)
            
            if resp.status_code == 200 and resp.json():
                row = resp.json()[0]
                row_id = row["id"]
                prob = row.get("model_prob", 50.0)

                # =========================================================
                # 5 TA SAVATGA SARALASH MANTIQI
                # =========================================================
                basket = "general_feedback"
                
                if reaction == "incorrect":
                    if prob >= 80.0 or prob <= 20.0:
                        basket = "quarantine_dispute"              # 1-savat: Zidlik / Karantin
                    elif 35.0 < prob < 65.0:
                        basket = "active_learning_uncertain_err"   # 2-savat: Oltin namuna (Ikkilanishdagi xato)
                    else:
                        basket = "borderline_disputed"
                elif reaction == "correct":
                    if prob >= 80.0 or prob <= 20.0:
                        basket = "verified_ground_truth"           # 3-savat: 100% tasdiqlangan bilim
                    elif 35.0 < prob < 65.0:
                        basket = "uncertain_confirmed"             # 4-savat: Ikkilanishdagi tasdiq
                    else:
                        basket = "general_confirmed"

                # Bazadagi qatorni yangilaymiz
                patch_url = f"{SUPABASE_URL}/rest/v1/threat_vault?id=eq.{row_id}"
                await client.patch(
                    patch_url, 
                    headers=SUPABASE_HEADERS, 
                    json={
                        "user_reaction": reaction,
                        "basket_type": basket
                    }
                )
                return {"status": "success", "assigned_basket": basket}
            else:
                raise HTTPException(status_code=404, detail="Xabar bazadan topilmadi.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
