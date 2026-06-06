from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("sut_model.joblib")

class SensorData(BaseModel):
    sicaklik: float
    nem: float
    amonyak: float
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)