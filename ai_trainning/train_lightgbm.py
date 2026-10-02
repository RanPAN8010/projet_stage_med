import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import joblib

# Ce script entraîne un modèle LightGBM pour classer l'état du conducteur, avec pondération équilibrée des classes.
# Il évalue les performances sur le jeu de test et exporte le modèle léger ainsi que le standardiseur au format .joblib.
def train_driver_status_lightgbm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_project_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(base_project_dir, 'data')
    ai_engine_med_dir = os.path.join(base_project_dir, 'ai_engine')
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    model_output_path = os.path.join(ai_engine_med_dir, 'lightgbm_body_model.joblib')
    scaler_output_path = os.path.join(ai_engine_med_dir, 'data_scaler_lightgbm.joblib')

    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}.")
        return

    print("Chargement de la base de données d'entraînement...")
    df = pd.read_csv(train_data_path)
    
    X = df[['HeartRate', 'HRV']]
    y = df['Label']
    
    # 划分训练集与测试集（保持 80/20 比例与分层抽样）
    # Diviser le jeu de données en ensemble d'entraînement et ensemble de test 
    # (maintien du ratio 80/20 et échantillonnage stratifié).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化
    # Normalisation des caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Entraînement du classifieur LightGBM en cours...")
    # class_weight='balanced': 自动根据样本比例调整权重
    # class_weight='balanced' : ajuster automatiquement les poids selon les proportions des classes pour gérer le déséquilibre.
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=100,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1  # verbose=-1 : désactiver l'affichage superflu des journaux d'itération.
    )
    
    # 开始训练
    # Lancer l'entraînement.
    model.fit(X_train_scaled, y_train)
    print("Entraînement du modèle terminé.")
    
    # 在测试集上进行预测并输出评估报告
    # Effectuer la prédiction sur le jeu de test et afficher le rapport d'évaluation.
    y_pred = model.predict(X_test_scaled)
    print("\n================ RAPPORT D'ÉVALUATION (LIGHTGBM) ================")
    print(classification_report(y_test, y_pred, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))
    print("==================================================================")
    
    # 确保输出目录存在
    # S'assurer que le répertoire de sortie existe.
    os.makedirs(ai_engine_med_dir, exist_ok=True)
    
    # 将模型与标准化工具保存至指定的 ai_engine/ 文件夹下
    # Sauvegarder le modèle et le standardiseur dans le dossier ai_engine/ spécifié.
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"Le modèle LightGBM et le standardiseur ont été enregistrés avec succès.")
    print(f"Modèle : {model_output_path}")
    print(f"Standardiseur : {scaler_output_path}")

if __name__ == "__main__":
    train_driver_status_lightgbm()