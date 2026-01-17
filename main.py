 

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
 
import tempfile
from pathlib import Path
 
import requests
import base64
import time
import json
from pydantic import BaseModel
import os
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import Column, Integer

from pydantic import BaseModel



class PaymentRequest(BaseModel):
    sender_email: str
    recipient_name: str
    recipient_phone: str
    recipient_email: str
    amount: str
    command: str
 

class Users(SQLModel, table=True):
    # Explicit autoincrement integer primary key
    # Let the sa_column define the primary key; do not pass primary_key to Field
    id: int | None = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    firstname: str = Field(index=True)
    lastname: str = Field(index=True)
    email: str = Field(index=True, unique=True)
    audio_sample_path: str | None = None
    face_image_path: str | None = None



app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


class UserCreate(SQLModel):
    firstname: str
    lastname: str
    email: str
    # Optional in the request; can be omitted or empty string
    audio_sample_path: str | None = None
    face_image_path: str | None = None


@app.post("/users/")
def create_user(user: UserCreate, session: SessionDep) -> Users:
    try:
        existing_user = session.exec(select(Users).where(Users.email == user.email)).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Normalize empty strings to None to avoid storing empty paths
    audio_path = user.audio_sample_path if (user.audio_sample_path and user.audio_sample_path.strip()) else None
    face_path = user.face_image_path if (user.face_image_path and user.face_image_path.strip()) else None

    new_user = Users(
        firstname=user.firstname,
        lastname=user.lastname,
        email=user.email,
        audio_sample_path=audio_path,
        face_image_path=face_path,
    )

    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user


@app.get("/users/{user_email}")
def read_user(user_email: str, session: SessionDep) -> Users:
    user = session.exec(select(Users).where(Users.email == user_email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/users/{user_email}/upload_audio")
async def upload_audio_for_user(user_email: str, session: SessionDep, file: UploadFile = File(...)):
    """Upload an audio sample for a user, save it to disk and store the path in the DB."""
    user = session.exec(select(Users).where(Users.email == user_email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Basic content-type validation
    if not (file.content_type and file.content_type.startswith("audio")):
        raise HTTPException(status_code=400, detail=f"Invalid file type for audio: {file.content_type}")

    base_dir = Path("user_files") / user_email
    base_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix or ".wav"
    filename = f"audio_{int(time.time())}{suffix}"
    dest = base_dir / filename

    # Save uploaded file
    with open(dest, "wb") as f:
        f.write(await file.read())

    # Update DB record
    user.audio_sample_path = str(dest)
    session.add(user)
    session.commit()
    session.refresh(user)

    return {"audio_sample_path": user.audio_sample_path}


@app.post("/users/{user_email}/upload_face")
async def upload_face_for_user(user_email: str, session: SessionDep, file: UploadFile = File(...)):
    """Upload a face image for a user, save it to disk and store the path in the DB."""
    user = session.exec(select(Users).where(Users.email == user_email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Basic content-type validation
    if not (file.content_type and file.content_type.startswith("image")):
        raise HTTPException(status_code=400, detail=f"Invalid file type for image: {file.content_type}")

    base_dir = Path("user_files") / user_email
    base_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix or ".jpg"
    filename = f"face_{int(time.time())}{suffix}"
    dest = base_dir / filename

    # Save uploaded file
    with open(dest, "wb") as f:
        f.write(await file.read())

    # Update DB record
    user.face_image_path = str(dest)
    session.add(user)
    session.commit()
    session.refresh(user)

    return {"face_image_path": user.face_image_path}

#simuler la connexion paypal
@app.post('/paypal_login/{user_email}')
def paypal_login(user_email: str, session: SessionDep):
    user = session.exec(select(Users).where(Users.email == user_email)).first()
    """if not user:
        raise HTTPException(status_code=404, detail="User not found")"""
    # Simuler la connexion PayPal
    paypal_token = "simulated_paypal_token_for_" + user_email
    return {"paypal_token": paypal_token}




@app.post('/paypal_pay')
def paypal_pay(payment: PaymentRequest, session: SessionDep):
    # Vérifier que l'utilisateur existe
    user = session.exec(select(Users).where(Users.email == payment.sender_email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Log de la transaction
    print(f"💰 Paiement de {payment.amount}€")
    print(f"   De: {payment.sender_email}")
    print(f"   À: {payment.recipient_name} ({payment.recipient_phone})")
    print(f"   Email destinataire: {payment.recipient_email}")
    print(f"   Commande: {payment.command}")
    
    # Simuler le paiement PayPal
    return {
        "status": "success",
        "message": f"Paiement de {payment.amount}€ envoyé à {payment.recipient_name}",
        "transaction_id": f"TXN_{payment.sender_email}_{payment.amount}",
        "sender": payment.sender_email,
        "recipient": payment.recipient_name,
        "amount": payment.amount
    }

 

@app.post("/face_id_photo/")
async def face_id_photo(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / file.filename
        with open(path, "wb") as f:
            f.write(await file.read())
        
       # analysis = DeepFace.analyze(img_path=str(path), actions=['age', 'gender', 'race', 'emotion'])   
    return "Save successful"


 
# PayPal config
client_id = "AYbHPzwV22hyJkj2zKlDD3Ji-LJERrMOEEQ2Ow3KhwENc4-pEB6zFZMpi2PSCIfeTwyiMHJK6AqiYRqR"
client_secret = "EH1rKafczFIApjFrPGTxscVfyD8_Xk6yqUsLwT73JC3IlxdOkcfhEPa8WSKp2KcnNDy8VWZvxSuzY3jW"
mode = "sandbox"   
base_url = f"https://api-m.{mode}.paypal.com"
redirect_uri = "http://127.0.0.1:8001/paypal_callback"  

# Scopes pour autoriser les payouts au nom de l'utilisateur (corrigé : /payments/payouts au lieu de /services/payouts)
scopes = "https://uri.paypal.com/services/paypalattributes https://uri.paypal.com/payments/payouts"

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




@app.post("/get_credentials")
async def get_credentials():
    
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "base_url": base_url
    }


