import streamlit as st
import pandas as pd
import catboost as cb
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Charger le modèle
model = cb.CatBoostClassifier()
model.load_model('meilleur_modele_catboost.cbm')

# Charger les données
data = pd.read_csv("application_train_preprocessed.csv")
data = data.fillna(0)
data_brut = pd.read_csv("application_train.csv")
data_brut = data_brut.fillna(0)

# Titre de l'application
st.title("Prédiction de solvabilité des clients")

# Quelques transformations de données
data_brut["DAYS_BIRTH"] = round(-data_brut["DAYS_BIRTH"]/365)
data_brut["DAYS_EMPLOYED"] = round(-data_brut["DAYS_EMPLOYED"]/12)
data_brut.loc[data_brut["FLAG_OWN_REALTY"] == "N", "FLAG_OWN_REALTY"] = "No"
data_brut.loc[data_brut["FLAG_OWN_REALTY"] == "Y", "FLAG_OWN_REALTY"] = "Yes"
data_brut.loc[data_brut["FLAG_OWN_CAR"] == "N", "FLAG_OWN_CAR"] = "No"
data_brut.loc[data_brut["FLAG_OWN_CAR"] == "Y", "FLAG_OWN_CAR"] = "Yes"


# Demander à l'utilisateur d'entrer l'ID du client
client_id = st.text_input("Entrez l'ID du client recherché :")

# Fonction d'affichage textuel d'informations clients
def aff_info(feature, phrase, unité=None):
    feature_client = client_data_brut[feature].values[0] if feature in client_data_brut.columns else "Non spécifié"
    if unité:
        st.write(phrase, f"{feature_client}", unité)
    else:
        st.write(phrase, f"{feature_client}")

if st.button("Prédire"):
    try:
        client_data = data[data['SK_ID_CURR'] == int(client_id)]
        client_data_brut = data_brut[data_brut['SK_ID_CURR'] == int(client_id)]

        if client_data.empty:
            st.error("Client non trouvé.")
        else:
            
            prediction = model.predict(client_data)
            solvable = bool(prediction[0])

            proba = model.predict_proba(client_data)
            probability = float(proba[0][1])
            
            # Afficher les informations sur le crédit demandé
            aff_info("AMT_CREDIT", "Montant du crédit demandé :", unité="$")
            aff_info("AMT_ANNUITY", "Montant de l'annuité :", unité="$")
            
            # Afficher les résultats
            #st.success(f"Client ID: {client_id}, Solvable: {solvable}, Score: {probability:.2f}")

            # Afficher la jauge pour le score
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                title={"text": "Score de Solvabilité"},
                gauge={
                    "axis": {"range": [0, 1], "tickcolor": "#FFFFFF"},
                    "bar": {"color": "#1E90FF"},
                    "steps": [
                        {"range": [0, 0.5], "color": "#FF4500"},
                        {"range": [0.5, 1], "color": "#32CD32"}
                    ],
                    "threshold": {
                        "line": {"color": "#FFD700", "width": 4},
                        "thickness": 0.75,
                        "value": 0.5
                    }
                }
            ))

            st.plotly_chart(gauge_fig)

            # Feature importance locale
            client_pool = cb.Pool(client_data)
            shap_values = model.get_feature_importance(client_pool, type="ShapValues")            
            shap_values_for_client = shap_values[0][:-1]  # On enlève la dernière colonne (base value)
            feature_names = data.columns
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values_for_client})            
            shap_df['Abs SHAP Value'] = shap_df['SHAP Value'].abs()
            top_positive = shap_df[shap_df['SHAP Value'] > 0].nlargest(10, 'SHAP Value')
            top_negative = shap_df[shap_df['SHAP Value'] < 0].nsmallest(10, 'SHAP Value')            
            top_shap_values = pd.concat([top_positive, top_negative])            
            fig = px.bar(
                top_shap_values, 
                x='SHAP Value', 
                y='Feature', 
                orientation='h',
                color_discrete_sequence=["#32CD32"],
                title="Elements clés du score du client")            
            st.plotly_chart(fig)
            
            #INFORMATIONS SUR LE DEMANDEUR
            st.subheader("Informations sur le demandeur")
            aff_info("CODE_GENDER", "Sexe :")
            aff_info("DAYS_BIRTH", "Age :")
            aff_info("NAME_EDUCATION_TYPE", "Dernier niveau de diplôme obtenu :")
            aff_info("NAME_HOUSING_TYPE", "Statut de l'habitationé :")

            #INFORMATIONS SUR LES REVENUS ET L EMPLOI
            st.subheader("Informations sur les revenus et l'emploi")
            aff_info("NAME_INCOME_TYPE", "Statut activité :")
            aff_info("OCCUPATION_TYPE", "Type de métier :")
            aff_info("ORGANIZATION_TYPE", "Secteur activité :")
            aff_info("DAYS_EMPLOYED", "Nombre de mois en activité :")            
            aff_info("AMT_INCOME_TOTAL", "Revenus totaux :","$")

            #Distribution revenus totaux
            data_filtered = data_brut[(data_brut['AMT_INCOME_TOTAL'] > 0)&(data_brut['AMT_INCOME_TOTAL'] < 350000)]
            fig = px.box(
                data_filtered,
                x='AMT_INCOME_TOTAL',
                title='Positionnement du client dans la distribution des revenus totaux',
                labels={'AMT_INCOME_TOTAL': 'Revenus Totaux ($)'},
            )

            revenus_totaux = client_data_brut["AMT_INCOME_TOTAL"].values[0]
            if revenus_totaux is not None:
                fig.add_shape(
                    type='line',
                    x0=revenus_totaux, x1=revenus_totaux,
                    y0=-0.5, y1=0.5,
                    line=dict(color='red', width=3, dash='dash'),
                )            
            
            fig.update_traces(marker=dict(color='#1E90FF'), fillcolor="#32CD32")            
            fig.update_layout(
                xaxis_title='Revenus Totaux ($)',
            )
            st.plotly_chart(fig)
            
            #PATRIMOINE
            st.subheader("Informations sur le patrimoine")
            aff_info("FLAG_OWN_REALTY", "Le demandeur est-il propriétaire d'un logement ?")
            aff_info("FLAG_OWN_CAR", "Le demandeur est-il propriétaire d'une voiture ?")

            #FAMILLE
            st.subheader("Informations sur la famille")
            aff_info("CNT_CHILDREN", "Nombre d\'enfants :")
            aff_info("NAME_FAMILY_STATUS", "Statut familial :")
            aff_info("CNT_FAM_MEMBERS", "Nombre de membres du foyer fiscal :")

            #LIEU DE VIE
            st.subheader("Informations sur le lieu de vie")

            aff_info("REGION_POPULATION_RELATIVE", "Densité du lieu de vie :")
            aff_info("REGION_RATING_CLIENT", "Notation de la région du lieu de vie :")
            aff_info("REGION_RATING_CLIENT_W_CITY", "Notation de la région du lieu de vie incluant la ville de référence :")

    except ValueError:
        st.error("Veuillez entrer un ID valide.")
    except Exception as e:
        st.error(f"Erreur : {str(e)}")

# Afficher les colonnes disponibles
st.write("Liste des données clients")
st.write(data_brut.columns.tolist())