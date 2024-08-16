from fastapi import FastAPI, HTTPException
import pandas as pd
import catboost as cb

app = FastAPI()

# Charger le modèle
model = cb.CatBoostClassifier()
model.load_model('meilleur_modele_catboost.cbm')

# Exemple de fonction pour obtenir les données client à partir de l'ID
def get_client_data(client_id: int) -> pd.DataFrame:
    # Remplacez cette partie par le chargement de vos données
    # Par exemple, à partir d'une base de données ou d'un fichier CSV
    # Exemple :
    df = pd.read_csv("../data/application_train_preprocessed.csv")
    client_data = df[df['SK_ID_CURR'] == client_id]
    
    return client_data

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.get("/predict/{client_id}")
def predict(client_id: int):
    try:
        # Récupérer les données du client
        client_data = get_client_data(client_id)
        
        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client not found")

        # Prédire la solvabilité
        prediction = model.predict(client_data)
        
        # Retourner le résultat
        solvable = bool(prediction[0])
        return {"client_id": client_id, "solvable": solvable}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
