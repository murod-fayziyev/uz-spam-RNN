from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import os
import re
import pickle
import hashlib
import httpx

# =====================================================================
# 1. KONFIGURATSIYA VA SUPABASE SOZLAMALARI
# =====================================================================
SUPABASE_URL = "https://hmmdrxeotfjscwudsyhb.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY or ''}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

app = FastAPI(
    title="Uzbek Cyber Threat & SOC Intelligence API",
    version="4.2 Pro",
    description="Bi-LSTM Attention + Data Hygiene Filter + Supabase Threat Vault"
)

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
        scores = self.attn(lstm_output).squeeze(2)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)
        context = torch.sum(lstm_output * weights, dim=1)
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
# 3. MODEL VA LUG'ATNI YUKLASH
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
        if os.path.exists(model_path) and vocab:
            model = BiLSTMClassifier(vocab_size=len(vocab))
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
    except Exception as e:
        print(f" [OGOHLANTIRISH] Model yuklashda xatolik: {e}")

load_ai_engine()

# =====================================================================
# 4. MATN TOZALIGI VA TASODIFIY BELGILAR FILTRI (ANTI-GIBBERISH)
# =====================================================================
VOWELS = set("aeiouyўаеёиоуыэюяAEIOUYЎАЕЁИОУЫЭЮЯ")
KEYBOARD_SMASH_PATTERNS = [
    "qwert", "asdfg", "zxcvb", "yuiop", "hjkl", 
    "йцуке", "фывап", "ячсми", "1234567"
]

def validate_meaningful_text(text: str):
    """
    Matn kamida 3 ta haqiqiy so'zdan iborat ekanligini va 
    tasodifiy harflar/belgilar yig'indisi emasligini tekshiradi.
    """
    cleaned = text.strip()
    if len(cleaned) < 10:
        return False, "Matn juda qisqa. Tahlil uchun to'liqroq xabar kiriting."

    # 1. URL havolalarini vaqtincha chetga olib, so'zlarni ajratish
    text_without_urls = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)
    words = re.findall(r"[a-zA-Zа-яА-ЯўқғҳЎҚҒҲёЁ]+(?:['ʻʼ`][a-zA-Zа-яА-ЯўқғҳЎҚҒҲёЁ]+)?", text_without_urls)
    
    # Kamida 2 harfli haqiqiy so'zlar ro'yxati
    valid_words = [w for w in words if len(w) >= 2]
    if len(valid_words) < 3:
        return False, "Xabar kamida 3 ta haqiqiy so'zdan iborat bo'lishi shart (qisqa matnlar bazaga yozilmaydi)."

    # 2. Klaviatura qatorlari (qwerty, asdfg) va 4+ bir xil ketma-ket harflarni tekshirish
    lower_text = text_without_urls.lower()
    for pat in KEYBOARD_SMASH_PATTERNS:
        if pat in lower_text:
            return False, "Matnda klaviaturada tasodifiy terilgan harflar (keyboard smash) aniqlandi."

    if re.search(r'([a-zA-Zа-яА-Яўқғҳё])\1{3,}', lower_text):
        return False, "Matnda ma'nosiz takrorlanuvchi harflar ketma-ketligi aniqlandi."

    # 3. Harflar va belgilar nisbati (faqat simvol/raqamlardan iborat emasligi)
    non_space_chars = re.sub(r'\s+', '', cleaned)
    all_letters = re.findall(r'[a-zA-Zа-яА-ЯўқғҳЎҚҒҲёЁ]', cleaned)
    if len(all_letters) / max(len(non_space_chars), 1) < 0.40:
        return False, "Matnda harflardan ko'ra tasodifiy belgilar yoki raqamlar haddan tashqari ko'p."

    # 4. Unli harflar tekshiruvi (har bir haqiqiy gapda unli harfli so'zlar bo'lishi shart)
    words_with_vowels = sum(1 for w in valid_words if any(ch in VOWELS for ch in w))
    if words_with_vowels < 3 or (words_with_vowels / len(valid_words)) < 0.60:
        return False, "Kiritilgan so'zlar ma'nosiz harflar yig'indisidan (gibberish) iborat."

    # Umumiy unlilar nisbati (12% dan 78% gacha tabiiy til me'yori)
    vowel_count = sum(1 for ch in all_letters if ch in VOWELS)
    vowel_ratio = vowel_count / max(len(all_letters), 1)
    if vowel_ratio < 0.12 or vowel_ratio > 0.78:
        return False, "Matn tabiiy til qoidalariga mos kelmaydi (tasodifiy harflar ketma-ketligi)."

    # Haddan tashqari uzun ma'nosiz so'z (masalan 28 ta harf ketma-ket bo'shliqsiz)
    if any(len(w) > 28 for w in valid_words):
        return False, "Matnda g'ayritabiiy uzunlikdagi tasodifiy harflar birikmasi mavjud."

    return True, "OK"

# =====================================================================
# 5. PII NIQOBLASH, IoC AJRATISH VA SOC MASLAHATCHI MANTIQI
# =====================================================================
def mask_personal_data(text: str) -> str:
    masked = re.sub(
        r'\b(8600|9860|5614|6262|4\d{3}|5\d{3})[\s\-]?(\d{2})\d{2}[\s\-]?\d{4}[\s\-]?(\d{4})\b',
        r'\1 \2** **** \3',
        text
    )
    masked = re.sub(
        r'(\+?998[\s\-]?[0-9]{2})[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}([\s\-]?[0-9]{2})',
        r'\1 *** ** \2',
        masked
    )
    return masked

def extract_threat_iocs(text: str):
    lower = text.lower()
    
    card_type = None
    if re.search(r'\b8600', text):
        card_type = "Uzcard (8600)"
    elif re.search(r'\b9860', text):
        card_type = "Humo (9860)"
    elif re.search(r'\b(5614|6262|4\d{3}|5\d{3})', text):
        card_type = "Bank kartasi"

    domain_info = None
    url_match = re.search(
        r'https?://([^/\s]+)|www\.([^/\s]+)|\b([a-zA-Z0-9\.\-]+\.(?:top|xyz|link|site|biz|ru|com|uz|info|online|org|net|cc))\b', 
        text, re.I
    )
    if url_match:
        extracted = url_match.group(1) or url_match.group(2) or url_match.group(3)
        domain_info = extracted.lower()

    has_apk = bool(".apk" in lower)
    has_otp_request = bool(re.search(r'\b(kod|sms|parol|tasdiqlash|cvv)\b', lower))

    return card_type, domain_info, has_apk, has_otp_request

def build_soc_intelligence(prob: float, card_type: str, domain: str, has_apk: bool, has_otp: bool):
    recommendations = []
    
    if has_apk:
        recommendations.append("Ushbu xabardagi '.apk' (o'rnatuvchi) faylni aslo yuklab olmang va o'rnatmang — u telefoningizdagi SMS va bank ilovalarini boshqaruvchi troyan bo'lishi mumkin.")
        recommendations.append("Agar faylni o'rnatgan bo'lsangiz, darhol internetni o'chiring, telefon Sozlamalari -> Maxsus ruxsatlar (Accessibility) bo'limidan shubhali ilovani o'chirib tashlang.")
    
    if card_type:
        recommendations.append(f"Xabarda {card_type} rekvizitlari qayd etilgan. Karta amal qilish muddati va SMS orqali kelgan bir martalik kodni hech kimga bermang.")
        recommendations.append("Agar karta ma'lumotlarini shubhali saytga kiritgan bo'lsangiz, darhol bank ilovasi orqali kartani 'Bloklash' (Muzlatish) tugmasini bosing.")

    if domain:
        is_suspicious_tld = any(domain.endswith(tld) for tld in [".top", ".xyz", ".site", ".link", ".online", ".biz", ".cc"])
        if is_suspicious_tld:
            recommendations.append(f"Diqqat: '{domain}' xavfli domen zonasida joylashgan! O'zbekistondagi rasmiy davlat va bank organlari faqat '.uz' yoki '.gov.uz' domenlaridan foydalanadi.")
        else:
            recommendations.append(f"Xabardagi '{domain}' havolasiga o'tishdan oldin uning haqiqiy rasmiy manzil ekanligini brauzerda qo'lda tekshiring.")

    if has_otp and not card_type:
        recommendations.append("Xabarda tasdiqlash kodi yoki parol so'ralmoqda. Telegram yoki bankdan kelgan 5-6 xonali kodni begona shaxslarga yoki botlarga yubormang.")

    if not recommendations:
        if prob >= 70.0:
            recommendations.append("Xabar matnida ijtimoiy muhandislik (aldov va shoshiltirish) alomatlari kuchli. Hech qanday shaxsiy ma'lumot yubormang.")
            recommendations.append("Yuboruvchining profilini bloklang va xabar ustiga bosib 'Spam' sifatida shikoyat (Report) qiling.")
        elif prob >= 35.0:
            recommendations.append("Xabar shubhali ko'rinmoqda. Undagi ma'lumotni tashkilotning rasmiy ishonch telefoni orqali qayta aniqlashtiring.")
        else:
            recommendations.append("Xabar xavfsiz ko'rinmoqda, biroq kiber-gigiyena qoidalariga amal qiling va shaxsiy kodlarni oshkor etmang.")

    if has_apk:
        category = "Mobile Malware / Banking Trojan Dropper"
        vector = "Zararli APK fayl tarqatish (T1404)"
    elif card_type and domain:
        category = "Financial Phishing & Card Harvesting"
        vector = "Soxta to'lov sahifasi orqali karta o'g'irlash (T1566)"
    elif domain:
        category = "Credential Phishing / Malicious Link"
        vector = "Shubhali tashqi havolaga yo'naltirish"
    elif prob >= 65.0:
        category = "Social Engineering / Scam Message"
        vector = "Matnli psixologik manipulyatsiya"
    elif prob >= 35.0:
        category = "Grayzone / Anomalous Content"
        vector = "Noaniq kontekst (Active Learning obyekti)"
    else:
        category = "Benign / Legitimate Traffic"
        vector = "Me'yoriy axborot xabari"

    return recommendations, {
        "threat_category": category,
        "attack_vector": vector
    }

def calculate_text_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

# =====================================================================
# 6. SUPABASE VAULT BILAN ISHLASH
# =====================================================================
async def sync_with_supabase(clean_text: str, text_hash: str, prob: float, card_type: str, domain: str, has_apk: bool):
    stats = {
        "occurrence_count": 1,
        "correct_count": 0,
        "incorrect_count": 0,
        "basket_type": "inference_raw_logs"
    }
    if not SUPABASE_KEY:
        return stats

    try:
        async with httpx.AsyncClient(timeout=4.5) as client:
            check_url = f"{SUPABASE_URL}/rest/v1/threat_vault?text_hash=eq.{text_hash}&select=*"
            resp = await client.get(check_url, headers=SUPABASE_HEADERS)
            
            if resp.status_code == 200 and resp.json():
                row = resp.json()[0]
                row_id = row["id"]
                new_occ = (row.get("occurrence_count") or 1) + 1
                
                patch_url = f"{SUPABASE_URL}/rest/v1/threat_vault?id=eq.{row_id}"
                await client.patch(patch_url, headers=SUPABASE_HEADERS, json={"occurrence_count": new_occ})
                
                stats["occurrence_count"] = new_occ
                stats["correct_count"] = row.get("correct_count") or 0
                stats["incorrect_count"] = row.get("incorrect_count") or 0
                stats["basket_type"] = row.get("basket_type") or "inference_raw_logs"
            else:
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
        print(f" [SUPABASE XATOLIGI]: {e}")

    return stats

# =====================================================================
# 7. API ENDPOINTLARI
# =====================================================================
class MessageRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text_hash: str
    user_reaction: str

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Uzbek Cyber Threat & SOC Intelligence API v4.2",
        "model_loaded": model is not None,
        "vault_configured": bool(SUPABASE_KEY)
    }

@app.post("/predict")
async def predict_message(req: MessageRequest):
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Bo'sh matn kiritildi.")

    # 1. Ma'lumot tozaligi va 3 ta so'z chegarasini tekshirish
    is_valid, reason = validate_meaningful_text(raw_text)
    if not is_valid:
        # Bazaga umuman yozilmaydi va foydalanuvchiga sababi qaytariladi
        raise HTTPException(status_code=422, detail=reason)

    text_hash = calculate_text_hash(raw_text)
    clean_text = mask_personal_data(raw_text)
    card_type, domain, has_apk, has_otp = extract_threat_iocs(raw_text)

    prob = 15.0
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
            basis = "Bi-LSTM + Attention Neyron Modeli"
    else:
        if has_apk or (card_type and domain):
            prob = 91.5
            basis = "Evristik Kiber-Qoidalar"

    recommendations, soc_info = build_soc_intelligence(prob, card_type, domain, has_apk, has_otp)
    
    # Faqat validatsiyadan o'tgan toza xabargina Supabase bazasiga yoziladi
    vault_stats = await sync_with_supabase(clean_text, text_hash, prob, card_type, domain, has_apk)

    return {
        "threat_probability": prob,
        "is_threat": prob >= 50.0,
        "text_hash": text_hash,
        "masked_text": clean_text,
        "decision_basis": basis,
        "detected_iocs": {
            "card": card_type,
            "domain": domain,
            "has_apk": has_apk,
            "has_otp": has_otp
        },
        "recommendations": recommendations,
        "soc_analysis": soc_info,
        "vault_stats": vault_stats
    }

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    reaction = req.user_reaction.lower()
    if reaction not in ["correct", "incorrect"]:
        raise HTTPException(status_code=400, detail="Noto'g'ri reaksiya turi.")

    try:
        async with httpx.AsyncClient(timeout=4.5) as client:
            url = f"{SUPABASE_URL}/rest/v1/threat_vault?text_hash=eq.{req.text_hash}&select=*"
            resp = await client.get(url, headers=SUPABASE_HEADERS)
            
            if resp.status_code == 200 and resp.json():
                row = resp.json()[0]
                row_id = row["id"]
                prob = row.get("model_prob", 50.0)
                
                correct_c = (row.get("correct_count") or 0) + (1 if reaction == "correct" else 0)
                incorrect_c = (row.get("incorrect_count") or 0) + (1 if reaction == "incorrect" else 0)

                if incorrect_c > correct_c:
                    if prob >= 80.0 or prob <= 20.0:
                        basket = "quarantine_dispute"
                    else:
                        basket = "active_learning_uncertain_err"
                else:
                    if prob >= 80.0 or prob <= 20.0:
                        basket = "verified_ground_truth"
                    else:
                        basket = "uncertain_confirmed"

                patch_url = f"{SUPABASE_URL}/rest/v1/threat_vault?id=eq.{row_id}"
                patch_payload = {
                    "user_reaction": reaction,
                    "basket_type": basket,
                    "correct_count": correct_c,
                    "incorrect_count": incorrect_c
                }
                patch_resp = await client.patch(patch_url, headers=SUPABASE_HEADERS, json=patch_payload)
                
                if patch_resp.status_code >= 400:
                    await client.patch(
                        patch_url, 
                        headers=SUPABASE_HEADERS, 
                        json={"user_reaction": reaction, "basket_type": basket}
                    )

                return {
                    "status": "success",
                    "assigned_basket": basket,
                    "correct_count": correct_c,
                    "incorrect_count": incorrect_c
                }
            else:
                raise HTTPException(status_code=404, detail="Xabar bazadan topilmadi.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
