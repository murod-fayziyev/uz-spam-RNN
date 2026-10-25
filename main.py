import os, re, pickle, hashlib
import httpx, torch, torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Render > Environment: SUPABASE_SERVICE_KEY (service_role kaliti!), VOTER_SALT (tasodifiy uzun matn)
SB_URL = os.getenv("SUPABASE_URL", "https://hmmdrxeotfjscwudsyhb.supabase.co")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SALT = os.getenv("VOTER_SALT", "change-me")
HDR = {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}", "Content-Type": "application/json"}
MAX_TEXT, MAX_LEN = 1500, 100

app = FastAPI(title="Uzbek Scam Checker", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

# ---------------- Model (o'zgarmagan) ----------------
class Attention(nn.Module):
    def __init__(self, h):
        super().__init__(); self.attn = nn.Linear(h * 2, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(2), dim=1).unsqueeze(2)
        return torch.sum(x * w, dim=1)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return torch.sigmoid(self.fc(self.attention(out)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vocab = None
try:
    with open("vocab_pro.pkl", "rb") as f:
        vocab = pickle.load(f)
    path = "best_spam_bilstm_pro.pt" if os.path.exists("best_spam_bilstm_pro.pt") else "best_model.pth"
    model = BiLSTMClassifier(len(vocab))
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
except Exception as e:
    print("Model yuklanmadi:", e)

def ml_score(text):
    if not (model and vocab):
        return 15.0
    ids = [vocab.get(w, 1) for w in re.findall(r"\b\w+\b", text.lower())][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    with torch.no_grad():
        return round(model(torch.tensor([ids]).to(device)).item() * 100, 1)

# ---------------- Matn yordamchilari ----------------
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def validate(text):
    words = re.findall(r"[^\W\d_]{2,}", re.sub(r"https?://\S+", " ", text))
    if len(words) < 3:
        return "Kamida 3 ta so'z kiriting."
    low = text.lower()
    if any(p in low for p in ("qwert", "asdfg", "zxcvb", "йцуке")) or re.search(r"([^\W\d_])\1{4,}", low):
        return "Matn ma'nosiz harflardan iborat ko'rinadi."

def mask(text):
    t = re.sub(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", "[KARTA]", text)
    t = re.sub(r"\+?998[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}", "[TELEFON]", t)
    t = re.sub(r"@\w{3,}", "[USERNAME]", t)
    t = re.sub(r"(https?://[^/\s]+)\S*", r"\1", t)   # havolada faqat domen qoladi
    return re.sub(r"\b\d{5,}\b", "[RAQAM]", t)

# ---------------- Ball asosidagi qoidalar ----------------
TLDS = "top|xyz|link|site|biz|ru|com|uz|info|online|org|net|cc|vip|click|live|shop"
BAD_TLD = tuple("." + t for t in "top xyz site link online biz cc vip click live shop".split())
TRUSTED = ("uzcard.uz", "humocard.uz", "payme.uz", "click.uz", "uzum.uz", "kapitalbank.uz", "nbu.uz",
           "cbu.uz", "anorbank.uz", "tbcbank.uz", "ipakyulibank.uz", "agrobank.uz", "xalqbanki.uz",
           "hamkorbank.uz", "gov.uz", "soliq.uz", "lex.uz")
BRAND = r"\b(soliq|mib|uzcard|humo|payme|click|uzum|bank|davlat|kapital|xalq|agro|hamkor|anor|ipak)"
CODE = r"\b(kod|parol|cvv|pin|sms)\w*"
ASK = r"\b(yuboring|ayting|kiriting|bering|jo'nating|yozing)\b"   # "bermang" ni ushlamaydi
CTA = r"\b(kiring|bosing|o'ting|tasdiqlang|yuklab oling)\b"
URG = r"zudlik|darhol|tezda|24 soat|bloklan|to'xtatil|muddati tugay|oxirgi imkoniyat|shoshilinch"
LURE = r"yutuq|yutdingiz|sovrin|bonus|keshbek|kompensatsiya|bepul"

ACTIONS = {
    "apk": "Faylni o'rnatmang. O'rnatgan bo'lsangiz, internetni o'chiring va ilovani Sozlamalar orqali o'chiring.",
    "card": "Karta ma'lumotini hech qayerga kiritmang. Kiritgan bo'lsangiz, bank ilovasidan kartani darrov bloklang.",
    "domain": "Havolaga kirmang. Xizmatni faqat rasmiy ilova yoki saytni o'zingiz ochib tekshiring (masalan soliq.uz, my.gov.uz).",
    "otp": "SMS kodni, parolni va CVV ni hech kimga aytmang, hatto bank xodimi desa ham.",
}

def analyze(text):
    low = re.sub(r"[ʻʼ’`‘]", "'", text.lower())
    m = re.search(r"https?://([^/\s]+)|\b((?:[a-z0-9-]+\.)+(?:%s))\b" % TLDS, low)
    domain = ((m.group(1) or m.group(2)).replace("www.", "").rstrip(".,)")) if m else None
    trusted = bool(domain) and any(domain == d or domain.endswith("." + d) for d in TRUSTED)
    card = re.search(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", text)
    card_type = None
    if card:
        d = re.sub(r"\D", "", card.group())[:4]
        card_type = "Uzcard" if d == "8600" else "Humo" if d == "9860" else "Bank kartasi"
    apk = ".apk" in low
    S = []  # (ball, sabab, kalit)
    if apk: S.append((60, ".apk fayl yuklashga undalmoqda", "apk"))
    if card: S.append((35, "Xabarda bank karta raqami bor", "card"))
    if domain and not trusted:
        bad = domain.endswith(BAD_TLD)
        S.append((35 if bad else 15, f"{domain} rasmiy sayt emas" + (" va xavfli domen zonasida" if bad else ""), "domain"))
        if re.search(BRAND, low): S.append((30, "Tashkilot nomi bor, lekin havola uning rasmiy sayti emas", "brand"))
        if re.search(CTA, low): S.append((10, "Havolaga o'tishga undalmoqda", "cta"))
    if re.search(CODE, low) and re.search(ASK, low): S.append((45, "Tasdiqlash kodi yoki parol so'ralmoqda", "otp"))
    if re.search(URG, low): S.append((10, "Shoshiltirish yoki qo'rqitish bor", "urgency"))
    if re.search(LURE, low): S.append((10, "Pul, bonus yoki yutuq va'da qilinmoqda", "lure"))
    official = bool(re.search(r"hech kim", low) and re.search(CODE, low)) and not (domain or card or apk or re.search(ASK, low))
    return S, domain, card_type, official

def category(keys, verdict):
    for k, name in (("apk", "Zararli dastur (.apk)"), ("brand", "Soxta tashkilot sayti"), ("otp", "Kod yoki parol undirish"),
                    ("card", "Karta ma'lumotini undirish"), ("lure", "Soxta yutuq yoki bonus"), ("domain", "Shubhali havola")):
        if k in keys: return name
    return "Shubhali xabar" if verdict != "xavfsiz" else "Oddiy xabar"

# ---------------- Supabase ----------------
async def rpc(name, body):
    async with httpx.AsyncClient(timeout=8) as c:
        return await c.post(f"{SB_URL}/rest/v1/rpc/{name}", headers=HDR, json=body)

async def save(args):
    """Muvaffaqiyatli bo'lsa xabar necha marta uchraganini, aks holda None qaytaradi."""
    try:
        r = await rpc("upsert_message", args)
        if r.status_code < 300:
            return int(r.json())
        print("Saqlash xatosi:", r.status_code, r.text[:300])
    except Exception as e:
        print("Saqlash xatosi:", e)
    return None

# ---------------- Endpointlar ----------------
class MessageRequest(BaseModel):
    text: str

class Vote(BaseModel):
    text_hash: str
    vote: str

@app.get("/")
def home():
    return {"status": "online", "version": "5.0", "model_loaded": model is not None, "vault": bool(SB_KEY)}

@app.post("/predict")
async def predict(req: MessageRequest, bg: BackgroundTasks):
    text = req.text.strip()
    if len(text) > MAX_TEXT:
        raise HTTPException(422, f"Matn {MAX_TEXT} belgidan oshmasin.")
    err = validate(text)
    if err:
        raise HTTPException(422, err)
    S, domain, card_type, official = analyze(text)
    ml = ml_score(text)
    rule = min(98, sum(s[0] for s in S))
    final = min(ml, 10.0) if official else max(ml, float(rule))
    verdict = "xavfli" if final >= 70 else "shubhali" if final >= 30 else "xavfsiz"
    keys = [s[2] for s in S]
    actions = [ACTIONS[k] for k in ("apk", "card", "domain", "otp") if k in keys]
    actions.append("Jo'natuvchini bloklang va xabarni spam deb belgilang." if verdict != "xavfsiz"
                   else "Xavfli belgilar topilmadi. Baribir kod va parollarni hech kimga bermang.")
    masked = mask(text)
    h = sha(" ".join(masked.lower().split()))
    seen = await save({"p_hash": h, "p_text": masked, "p_ml": ml, "p_final": final, "p_verdict": verdict,
                       "p_signals": keys, "p_card": card_type, "p_domain": domain, "p_apk": "apk" in keys})
    return {"verdict": verdict, "score": round(final, 1), "category": category(keys, verdict),
            "reasons": [s[1] for s in S], "actions": actions, "text_hash": h,
            "saved": seen is not None, "seen": seen,
            "details": {"model": ml, "rules": rule, "domain": domain, "card": card_type}}

@app.post("/feedback")
async def feedback(v: Vote, request: Request):
    if v.vote not in ("spam", "safe", "unsure") or not re.fullmatch(r"[0-9a-f]{64}", v.text_hash):
        raise HTTPException(400, "Noto'g'ri so'rov.")
    cid = request.headers.get("x-client-id", "")[:64]
    if len(cid) < 8:
        raise HTTPException(400, "Brauzer identifikatori yo'q.")
    ip = (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()
    r = await rpc("cast_vote", {"p_hash": v.text_hash, "p_voter": sha(SALT + cid + ip),
                                "p_ip": sha(SALT + ip), "p_vote": v.vote})
    if r.status_code >= 400:
        if "limit" in r.text: raise HTTPException(429, "Bugungi limit tugadi (kuniga 20 ta).")
        if "not_found" in r.text: raise HTTPException(404, "Xabar topilmadi. Qayta tekshirib ko'ring.")
        raise HTTPException(502, "Saqlashda xatolik.")
    return {"ok": True}


@app.get("/vault-check")
async def vault_check():
    """Baza ulanishini tekshirish: brauzerda /vault-check ni oching."""
    if not SB_KEY:
        return {"ok": False, "reason": "Render da SUPABASE_SERVICE_KEY o'rnatilmagan"}
    out = {}
    async with httpx.AsyncClient(timeout=8) as c:
        for name, q in (("votes_table", "feedback_votes?select=id&limit=1"),
                        ("vault_columns", "threat_vault?select=final_prob&limit=1")):
            r = await c.get(f"{SB_URL}/rest/v1/{q}", headers=HDR)
            out[name] = {"status": r.status_code, "error": None if r.status_code < 300 else r.text[:200]}
    out["ok"] = all(v["status"] < 300 for v in out.values())
    return out
