from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "sut_model.joblib"

SUT_API = "http://localhost:8080/api/sut-verim"

model = joblib.load(MODEL_PATH)

model_status = {
    "son_egitim": "-",
    "veri_sayisi": 0,
    "r2_skoru": None,
    "feature_importance": None,
    "durum": "Hazır"
}

class SensorData(BaseModel):
    sicaklik: float
    nem: float
    amonyak: float
    isik: float = 0
    yem_tuketimi: float = 6

@app.post("/predict")
def predict(data: SensorData):
    features = pd.DataFrame([{
        "sicaklik": data.sicaklik,
        "nem": data.nem,
        "amonyak": data.amonyak,
        "yem_tuketimi": data.yem_tuketimi
    }])

    prediction = model.predict(features)[0]

    durum = "Normal" if prediction >= 3 else "Düşük Verim Riski"

    return {
        "tahmin_edilen_sut": round(float(prediction), 2),
        "durum": durum
    }

@app.post("/train-live")
def train_live():
    global model
    global model_status

    try:
        sut_response = requests.get(SUT_API, timeout=10)

        if not sut_response.ok:
            return {
                "durum": "Hata",
                "mesaj": "Süt verileri alınamadı."
            }

        sut_data = sut_response.json()

        if len(sut_data) < 5:
            return {
                "durum": "Yetersiz Veri",
                "mesaj": "Model eğitimi için en az 5 kayıt gereklidir.",
                "veri_sayisi": len(sut_data)
            }

        rows = []

        for item in sut_data:

            if (
                item.get("sicaklik") is None or
                item.get("nem") is None or
                item.get("amonyak") is None
            ):
                continue

            rows.append({
                "sicaklik": float(item.get("sicaklik", 0)),
                "nem": float(item.get("nem", 0)),
                "amonyak": float(item.get("amonyak", 0)),
                "yem_tuketimi": float(item.get("yemTuketimi", 0)),
                "sut_verimi": float(item.get("sutVerimi", 0))
            })

        if len(rows) < 5:
            return {
                "durum": "Yetersiz Veri",
                "mesaj": "Sensör verisi içeren en az 5 kayıt gereklidir."
            }

        df = pd.DataFrame(rows)

        X = df[
            [
                "sicaklik",
                "nem",
                "amonyak",
                "yem_tuketimi"
            ]
        ]

        y = df["sut_verimi"]

        if len(df) >= 5:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            yeni_model = RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )

            yeni_model.fit(X_train, y_train)

            y_pred = yeni_model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

        else:

            yeni_model = RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )

            yeni_model.fit(X, y)

            r2 = None

        importance = yeni_model.feature_importances_

        feature_importance = {
            "sicaklik": round(float(importance[0] * 100), 1),
            "nem": round(float(importance[1] * 100), 1),
            "amonyak": round(float(importance[2] * 100), 1),
            "yem_tuketimi": round(float(importance[3] * 100), 1)
        }

        joblib.dump(yeni_model, MODEL_PATH)

        model = yeni_model

        model_status = {
            "son_egitim": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "veri_sayisi": len(df),
            "r2_skoru": round(float(r2), 2) if r2 is not None else None,
            "feature_importance": feature_importance,
            "durum": "Model güncellendi"
        }

        return model_status

    except Exception as e:
        return {
            "durum": "Hata",
            "mesaj": str(e)
        }

@app.get("/model-status")
def get_model_status():
    return model_status

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )