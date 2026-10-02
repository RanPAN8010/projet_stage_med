import os
import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model

# ==============================================================================
# [Description en français]
# Ce script utilise le framework AutoML PyCaret pour sélectionner et entraîner un modèle
# de classification sur les données d'état physiologique du conducteur.
# Il charge les données basées sur la fréquence cardiaque (HeartRate) et la variabilité
# de la fréquence cardiaque (HRV), traite le déséquilibre des classes, puis compare
# automatiquement plusieurs algorithmes en privilégiant le score F1.
# Enfin, le meilleur modèle est réentraîné sur l'intégralité du jeu de données et exporté
# avec son pipeline de prétraitement au format .pkl.
# ==============================================================================
def run_automl_medical_selection():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_project_dir = os.path.abspath(os.path.join(current_dir, '..'))

    data_dir = os.path.join(base_project_dir, 'data')
    ai_engine_med_dir = os.path.join(base_project_dir, 'ai_engine')
    
    train_data_path = os.path.join(data_dir, 'driver_body_status_train.csv')
    model_output_prefix = os.path.join(ai_engine_med_dir, 'best_automl_med_model')

    if not os.path.exists(train_data_path):
        print(f"Erreur : Base de données introuvable à l'emplacement {train_data_path}.")
        return

    print("Chargement de la base de données pour l'AutoML...")
    df = pd.read_csv(train_data_path)
    
    # 只提取二维生理特征和标签
    # Extraire uniquement les deux caractéristiques physiologiques et le label.
    data_for_automl = df[['HeartRate', 'HRV', 'Label']]
    
    print("\n================ CONFIGURATION DE L'EXPÉRIENCE ================")
    # 初始化 PyCaret 环境 Initialiser l'environnement PyCaret.
    # target: 预测的目标列 Colonne cible à prédire.
    # train_size: 训练集比例 (80/20 分割) Proportion de l'ensemble d'entraînement (répartition 80/20).
    # fix_imbalance: Activer automatiquement des algorithmes tels que SMOTE pour gérer 
    #                le déséquilibre lié aux échantillons rares de maladies cardiaques.
    # html=False: Assurer un affichage clair sous forme de texte brut dans le terminal/PowerShell
    clf_setup = setup(
        data=data_for_automl,
        target='Label',
        train_size=0.8,
        fix_imbalance=True,
        preprocess=True,
        numeric_features=['HeartRate', 'HRV'],
        session_id=42,
        verbose=True,
    )
    
    print("\n================ COMPARAISON DES MODÈLES D'IA ================")
    print("Évaluation automatique de tous les algorithmes disponibles (RF, KNN, XGBoost, etc.)...")
    
    # sort='F1': 鉴于医疗高危场景，我们让它优先以 F1-score 作为核心排序标准，平衡精确率与召回率
    # sort='F1' : Compte tenu du contexte médical à haut risque, 
    # le score F1 est utilisé comme critère principal de tri afin d'équilibrer précision et rappel
    # include: 显式指定重点考察对比的几个核心模型
    # include : Spécifier explicitement les modèles clés à évaluer,
    best_model = compare_models(
        exclude=['svm', 'gpc', 'rbfsvm'],
        sort='F1',
    )
    
    print("\n================ FINALISATION DU MEILLEUR MODÈLE ================")
    print("Entraînement final sur l'intégralité des données...")
    # 锁定并在全部数据上完整重新训练表现最好的那个模型
    # réentraîner entièrement le modèle le plus performant sur l'ensemble complet des données.
    final_model = finalize_model(best_model)
    
    # 确保输出目录存在
    # S'assurer que le répertoire de sortie existe.
    os.makedirs(ai_engine_med_dir, exist_ok=True)
    
    # 保存最优模型
    # Sauvegarder le meilleur modèle dans un fichier .pkl
    save_model(final_model, model_output_prefix)
    print(f"\nLe meilleur modèle a été enregistré avec succès sous : {model_output_prefix}.pkl")
    print("================================================================")

if __name__ == "__main__":
    run_automl_medical_selection()