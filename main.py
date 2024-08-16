from fastapi import FastAPI, HTTPException
import pandas as pd
import catboost as cb

app = FastAPI()

# Charger le modèle
model = cb.CatBoostClassifier()
model.load_model('meilleur_modele_catboost.cbm')
data = pd.read_csv("application_train_preprocessed.csv")
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

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
        
        return {"client_id": client_id, "solvable": solvable}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
