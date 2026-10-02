======================================================================
SYSTÈME EMBARQUÉ DE SURVEILLANCE PHYSIOLOGIQUE DU CONDUCTEUR
DRIVER PHYSIOLOGICAL MONITORING EMBEDDED SYSTEM
Présentation du projet

Ce projet constitue le sous-système d'analyse biomédicale pour la surveillance embarquée de l'état du conducteur. Le système exploite des capteurs physiologiques (oxymètre de pouls et moniteur de fréquence cardiaque) pour extraire en temps réel deux métriques physiologiques clés : la fréquence cardiaque (HeartRate) et la variabilité de la fréquence cardiaque (HRV).

Un modèle d'apprentissage automatique XGBoost entraîné classifie l'état de l'utilisateur en trois classes :

0 : Normal (Sain)

1 : Fatigue Mentale

2 : Crise Cardiaque (Anomalie aiguë)

Structure du projet

projet_stage_med/
ai_engine/          : Modèles entraînés et outils de normalisation (xgboost_body_model.joblib, data_scaler_xgboost.joblib)
ai_trainning/       : Scripts d'entraînement, d'optimisation et d'inférence médicale
data/               : Données physiologiques brutes (FatigueSet, Heart Statlog) et jeux d'entraînement
utils/
data_prep/      : Nettoyage, rééchantillonnage temporel et fusion des jeux de données
hardware/       : Scripts de scan et d'acquisition BLE pour capteurs médicaux
requirements.txt    : Liste des dépendances logicielles du projet

1. Collecte matérielle des données (Raspberry Pi & Capteur BLE)

Étape 1 : Connexion au Raspberry Pi
Branchez le câble réseau et l'alimentation sur le Raspberry Pi.
Sur votre ordinateur, ouvrez PuTTY et connectez-vous en SSH à l'adresse 10.3.183.6 (identifiant : pi, mot de passe : raspberry).

Étape 2 : Scan des périphériques Bluetooth
Sur le Raspberry Pi, entrez dans le dossier de travail : cd ~/IoT
Lancez la commande : python scan.py. Le programme commence à scanner les signaux Bluetooth environnants.

Étape 3 : Activation de l'oxymètre
Appuyez sur le petit bouton blanc de l'oxymètre de pouls pour l'activer.

Étape 4 : Récupération de l'adresse MAC
Dans la liste des appareils Bluetooth affichés dans le terminal, repérez celui nommé Libelium ou MySignals, puis notez son adresse MAC.

Étape 5 : Lancement de l'acquisition
Vérifiez que l'adresse MAC dans le fichier read_spo2.py correspond bien à l'adresse trouvée à l'étape précédente. Si ce n'est pas le cas, modifiez-la avec la bonne adresse.
Lancez ensuite la commande : python read_spo2.py. Le programme passe en attente de données.

Étape 6 : Mesure sur l'utilisateur
Activez l'oxymètre en appuyant sur son bouton blanc et pincez-le sur votre doigt.

Étape 7 : Enregistrement des données
Le Raspberry Pi enregistre automatiquement les mesures reçues dans le fichier med_realtime_inference_input.csv.

2. Préparation des données et entraînement de l'IA

Étape 1 : Préparation et fusion des jeux de données
Assurez-vous que les données brutes sont placées dans le dossier data/ :

https://www.kaggle.com/datasets/tanjemahamed/mental-fatigue-level-detection-fatigueset-data
data/fatigueset/ (Dossiers des participants 01 à 12)

https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final
data/heart_statlog_cleveland_hungary_final.csv

Exécutez le pipeline de nettoyage temporel et d'alignement des données :
python utils/data_prep/clean_fatigueset.py
-> Génère le fichier intermédiaire : data/fatigueset_cleaned.csv

python utils/data_prep/merge_heart_data.py
-> Génère le jeu d'entraînement final : data/driver_body_status_train.csv

Étape 2 : Entraînement du modèle XGBoost
Lancez l'entraînement du classifieur XGBoost avec pondération des classes :
python ai_trainning/train_xgboost.py
-> Fichiers exportés dans ai_engine/ :

xgboost_body_model.joblib

data_scaler_xgboost.joblib

Étape 3 : Évaluation et inférence
(Optionnel) Générer un jeu de données de test synthétique pour l'inférence :
python utils/data_prep/generate_mock_med_data.py
-> Génère : data/med_sensor_inference_test.csv

Lancer le service d'inférence médicale sur les données capteurs :
python ai_trainning/predict_med_service.py
-> Exporte les diagnostics prédictifs sous : data/med_inference_output.csv

(Optionnel) Visualiser les frontières de décision du modèle :
python ai_trainning/plot_model_boundaries.py