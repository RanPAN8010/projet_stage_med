import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Ce script entraîne un modèle de régression logistique pour classer l'état du conducteur, avec normalisation et équilibrage des classes.
# Il évalue ensuite les performances sur le jeu de test et sauvegarde le modèle ainsi que le standardiseur au format .joblib.
def train_driver_status_logistic_regression():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    ai_engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'ai_engine'))
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    model_output_path = os.path.join(ai_engine_dir, 'logistic_regression_model.joblib')
    scaler_output_path = os.path.join(ai_engine_dir, 'data_scaler_logistic.joblib')

    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}. Veuillez vérifier l'exécution de merge_heart_data.py.")
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
    
    # standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Entraînement du classifieur Régression Logistique en cours...")
    # 使用 class_weight='balanced' 来对抗严重的数据不平衡问题
    # class_weight='balanced' pour compenser le fort déséquilibre des classes.
    # multi_class='multinomial' 用于处理三分类任务
    # multi_class='multinomial' permet de traiter la tâche de classification à trois classes.
    model = LogisticRegression(
        class_weight='balanced',
        solver='lbfgs',
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    print("Entraînement du modèle terminé.")
    
    # 在测试集上进行预测并输出评估报告
    # Effectuer la prédiction sur le jeu de test et afficher le rapport d'évaluation.
    y_pred = model.predict(X_test_scaled)
    print("\n================ RAPPORT D'ÉVALUATION ================")
    print(classification_report(y_test, y_pred, target_names=['0:Normal', '1:Fatigue', '2:Crise_Cardiaque']))
    print("======================================================")
    
    # 将模型与标准化工具保存至指定的 ai_engine 文件夹下
    # Sauvegarder le modèle et le standardiseur dans le dossier ai_engine spécifié.
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"Le modèle Régression Logistique et le standardiseur ont été enregistrés avec succès dans le dossier ai_engine.")

if __name__ == "__main__":
    train_driver_status_logistic_regression()