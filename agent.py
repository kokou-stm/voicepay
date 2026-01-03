"""import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs =torchaudio.load('output1.wav')
embeddings = classifier.encode_batch(signal)

curl -X POST "http://127.0.0.1:8000/verify" \
-F "file1=@/Users/sekponakokou/Desktop/ownprojects/testvoiceid/output1.wav" \
-F "file2=@/Users/sekponakokou/Desktop/ownprojects/testvoiceid/output2.wav"

"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.inference.speaker import SpeakerRecognition
import tempfile
from pathlib import Path
from deepface import DeepFace
import requests
import base64
import time
import json
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

@app.post("/verify")
async def verify_speaker(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / file1.filename
        path2 = Path(tmpdir) / file2.filename
        
        with open(path1, "wb") as f1:
            f1.write(await file1.read())
        with open(path2, "wb") as f2:
            f2.write(await file2.read())
        
        score, prediction = verification.verify_files(str(path1), str(path2))
    
    return {
        "score": score.item(),
        "prediction": bool(prediction.item())
    }

@app.post("/verify_face")
async def verify_face(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / file1.filename
        path2 = Path(tmpdir) / file2.filename
        
        with open(path1, "wb") as f1:
            f1.write(await file1.read())
        with open(path2, "wb") as f2:
            f2.write(await file2.read())
        
        result = DeepFace.verify(img1_path=str(path1), img2_path=str(path2), model_name="ArcFace")  # Ou "SFace" pour plus de vitesse
    
    return result   

# PayPal config
client_id = "AYbHPzwV22hyJkj2zKlDD3Ji-LJERrMOEEQ2Ow3KhwENc4-pEB6zFZMpi2PSCIfeTwyiMHJK6AqiYRqR"
client_secret = "EH1rKafczFIApjFrPGTxscVfyD8_Xk6yqUsLwT73JC3IlxdOkcfhEPa8WSKp2KcnNDy8VWZvxSuzY3jW"
mode = "sandbox"  # or "live"
base_url = f"https://api-m.{mode}.paypal.com"
redirect_uri = "http://127.0.0.1:8001/paypal_callback"  # Remplacez par votre URL de callback (ajustez le port si nécessaire, e.g., 8001)

# Scopes pour autoriser les payouts au nom de l'utilisateur (corrigé : /payments/payouts au lieu de /services/payouts)
scopes = "openid email https://uri.paypal.com/services/paypalattributes https://uri.paypal.com/payments/payouts"

@app.get("/paypal_authorize")
async def paypal_authorize():
    auth_url = f"https://www.{mode}.paypal.com/signin/authorize?client_id={client_id}&response_type=code&scope={scopes.replace(' ', '%20')}&redirect_uri={redirect_uri}"
    return RedirectResponse(auth_url)

@app.get("/paypal_callback")
async def paypal_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Code d'autorisation manquant")
    
    # Échange code pour access token
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri
    }
    response = requests.post(f"{base_url}/v1/oauth2/token", headers=headers, data=data)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    tokens = response.json()
    # En prod, stockez les tokens sécurisé (e.g., DB avec user ID)
    print("Access Token:", tokens["access_token"])
    print("Refresh Token:", tokens.get("refresh_token"))  # Pour refresh plus tard
    
    return {"message": "Autorisation réussie ! Access token obtenu.", "tokens": tokens}

class PayoutRequest(BaseModel):
    amount: str  # e.g., "10.00"
    currency: str  # e.g., "EUR"
    receiver: str  # e.g., "sb-guwlq37791654@personal.example.com"
    access_token: str  # L'access token de l'utilisateur

@app.post("/create_payout")
async def create_payout(payout_req: PayoutRequest):
    payout_data = {
        "sender_batch_header": {"email_subject": "Transfert instantané"},
        "items": [{
            "recipient_type": "EMAIL",
            "amount": {"value": payout_req.amount, "currency": payout_req.currency},
            "receiver": payout_req.receiver
        }]
    }

    headers = {
        "Authorization": f"Bearer {payout_req.access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(f"{base_url}/v1/payments/payouts", headers=headers, json=payout_data)

    if response.status_code == 201:
        batch = response.json()
        print("Payout batch created successfully!")
        batch_id = batch["batch_header"]["payout_batch_id"]
        
        # Polling du statut (comme avant, avec retries augmentés)
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            status_response = requests.get(f"{base_url}/v1/payments/payouts/{batch_id}", headers=headers)
            if status_response.status_code == 200:
                status = status_response.json()
                batch_status = status["batch_header"]["batch_status"]
                print("Payout status:", batch_status)
                if batch_status in ['SUCCESS', 'DENIED', 'FAILED']:
                    return {"status": batch_status, "details": status}
            elif status_response.status_code == 404:
                print("Batch not found yet (404), retrying...")
            else:
                raise HTTPException(status_code=status_response.status_code, detail=status_response.text)
            
            time.sleep(10)
            retry_count += 1
        
        raise HTTPException(status_code=408, detail="Max retries reached; batch status could not be retrieved.")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)