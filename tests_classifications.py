import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

# Test de l'importation des données
def test_data_import():
    df = pd.read_csv("application_train_preprocessed.csv")
    assert not df.empty, "Le dataframe est vide."
    assert 'TARGET' in df.columns, "La colonne TARGET est manquante."

# Test de la division des données
def test_train_test_split():
    df = pd.read_csv("application_train_preprocessed.csv")
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert X_train.shape[0] == y_train.shape[0], "Problème dans la taille du jeu d entraînement"
    assert X_test.shape[0] == y_test.shape[0], "Problème dans la taille du jeu de test"
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Le split train/test a échoué"

# Test de la configuration MLFlow
def test_mlflow_experiment_setup():
    experiment_name = "Projet_7"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    assert experiment is not None, f"Experiment {experiment_name} not created!"
