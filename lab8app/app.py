from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Dict

app = FastAPI()

# Define the input data model
class DiamondInput(BaseModel):
    features: Dict[str, float]

# Load the model at startup
@app.on_event("startup")
async def load_model():
    try:
        # Set MLflow tracking URI to local server
        mlflow.set_tracking_uri("http://localhost:5001")
        
        # Load the model from MLflow
        model_uri = "models:/lab6-best-model/1"
        app.model = mlflow.sklearn.load_model(model_uri)
        
        # Define feature names in the correct order
        app.feature_names = ["y", "carat", "x", "z", "clarity_SI2", "clarity_I1", 
                           "color_J", "clarity_SI1", "color_I", "clarity_VVS2", 
                           "depth", "color_H"]
        
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.post("/predict")
async def predict(input_data: DiamondInput):
    try:
        # Convert input dictionary to DataFrame with correct feature order
        input_df = pd.DataFrame([input_data.features], columns=app.feature_names)
        
        # Make prediction
        prediction = app.model.predict(input_df)[0]
        
        return {"predicted_price": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 