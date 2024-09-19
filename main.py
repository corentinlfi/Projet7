from fastapi import FastAPI, HTTPException
import pandas as pd
import catboost as cb
import numpy as np

app = FastAPI()

# Charger le modèle
model = cb.CatBoostClassifier()
model.load_model('meilleur_modele_catboost.cbm')
data = pd.read_csv("application_train_preprocessed.csv")
data = data.fillna(0)

def get_client_data(client_id: int) -> pd.DataFrame:
    if 'SK_ID_CURR' not in data.columns:
        raise HTTPException(status_code=500, detail="Column 'SK_ID_CURR' not found in data")
    client_data = data[data['SK_ID_CURR'] == client_id]
    return client_data

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.get("/predict/{client_id}")
def predict(client_id: int):
    try:
        client_data = get_client_data(client_id)
        if client_data is None:
            raise HTTPException(status_code=404, detail="Client not found")
        
        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client not found")
        
        prediction = model.predict(client_data)
        solvable = bool(prediction[0])

        proba = model.predict_proba(client_data)
        probability = float(proba[0][1])

        """TEST AFFICHAGE DATAFRAME FEATURES CLIENT"""
        
        client_features = client_data.to_dict(orient="records")[0]

        client_features_df = {
            "feature": list(client_features.keys()),
            "value": list(client_features.values())
        }
        
        return {
            "client_id": client_id,
            "solvable": solvable,
            "probability": probability,
            "client_features": client_features_df
        }

        """FIN DU TEST"""
    
        #return {"client_id": client_id, "solvable": solvable, "probability": probability}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
