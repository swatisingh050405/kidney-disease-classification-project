from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import base64
import os

from kidney_disease_classifier.pipeline.prediction import PredictionPipeline


# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Templates folder
templates = Jinja2Templates(directory="templates")

# ✅ Load model once at startup
predictor = PredictionPipeline()


# ===============================
# Home Route (HTML Page)
# ===============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===============================
# Prediction Route
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # ✅ Read uploaded file
        image_bytes = await file.read()

        # ✅ Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # ✅ Get prediction
        result = predictor.predict(image_base64)

        return JSONResponse(content={
            "prediction": result["prediction"],
            "confidence": f"{result['confidence']}%"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)