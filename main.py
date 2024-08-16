from fastapi import FastAPI, HTTPException
import pandas as pd
import catboost as cb

app = FastAPI()

# Charger le modèle
model = cb.CatBoostClassifier()
model.load_model('meilleur_modele_catboost.cbm')

# Charger les données une seule fois pour éviter les frais de chargement récurrents
data = pd.read_csv("../data/application_train_preprocessed.csv")

def get_client_data(client_id: int) -> pd.DataFrame:
    # Filtrer les données pour obtenir les informations du client spécifique
    client_data = data[data['client_id'] == client_id]
    
    if client_data.empty:
        return None
    
    # Assurez-vous que les données sont prétraitées correctement
    # Exemples de prétraitement : normalisation, gestion des valeurs manquantes, etc.
    # Ajustez cette partie en fonction de votre modèle
    client_data = client_data.drop(columns=['client_id'])  # Exclure la colonne ID pour la prédiction

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
        
        # Prévoir en utilisant le modèle
        prediction = model.predict(client_data)
        solvable = bool(prediction[0])
        
        return {"client_id": client_id, "solvable": solvable}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
