import os
import pandas as pd
import numpy as np

def analyze_hrv_correlation():
    print("[Progression] Démarrage du script, calcul du chemin cible...")
    
    #Localisation dynamique de la racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
    data_path = os.path.join(base_project_dir, 'data', 'driver_body_status_train.csv')
    print(f"[Progression] Chemin physique du jeu d'entraînement :\n  -> {data_path}")
    
    if not os.path.exists(data_path):
        print("[Erreur] Fichier d'entraînement fusionné introuvable. Veuillez vérifier l'emplacement !")
        return
        
    print("[Progression] Données détectées, chargement en mémoire...")
    df = pd.read_csv(data_path)
    print(f"[Progression] Chargement réussi ! Nombre total de lignes : {len(df)}")
    
    #Extraction des valeurs de HRV selon les étiquettes
    hrv_fatigue = df[df['Label'] == 1]['HRV']
    hrv_heart_disease = df[df['Label'] == 2]['HRV']
    
    print(f"[Progression] Extraction terminée. Échantillons de fatigue : {len(hrv_fatigue)}, Crises cardiaques : {len(hrv_heart_disease)}")
    
    if len(hrv_fatigue) == 0 or len(hrv_heart_disease) == 0:
        print("[Avertissement] Le nombre d'échantillons pour l'une des classes est égal à zéro, comparaison impossible.")
        return

    print("\n========== ANALYSE DE CORRÉLATION DES VALEURS HRV ==========\n")
    
    # Comparaison statistique descriptive
    print("Comparaison des statistiques descriptives :")
    print(f"-> FatigueSet (Fatigue) - Plage HRV : [{hrv_fatigue.min():.2f} à {hrv_fatigue.max():.2f}], Moyenne : {hrv_fatigue.mean():.2f}")
    print(f"-> Heart Statlog (Pathologie) - Plage HRV : [{hrv_heart_disease.min():.2f} à {hrv_heart_disease.max():.2f}], Moyenne : {hrv_heart_disease.mean():.2f}")
    
    # Recherche de chevauchement physique des valeurs
    fatigue_rounded = np.round(hrv_fatigue, 1)
    heart_rounded = np.round(hrv_heart_disease, 1)
    
    intersection = np.intersect1d(fatigue_rounded, heart_rounded)
    print("\nTest d'intersection des valeurs numériques :")
    print(f"-> En arrondissant à 1 décimale, les deux jeux de données partagent {len(intersection)} valeurs de HRV identiques !")
    if len(intersection) > 0:
        print(f"-> Exemples de points communs (5 premiers) : {intersection[:5]}")
        
    #  Calcul du taux de recouvrement spatial
    min_f, max_f = hrv_fatigue.min(), hrv_fatigue.max()
    in_range_count = hrv_heart_disease.between(min_f, max_f).sum()
    overlap_percentage = (in_range_count / len(hrv_heart_disease)) * 100
    
    print("\nAnalyse du recouvrement spatial :")
    print(f"-> {overlap_percentage:.1f}% des valeurs de HRV du jeu cardiologique se situent dans la plage de valeurs du jeu de fatigue.")
    print("\nConclusion :")
    if overlap_percentage > 50:
        print("[Corrélation confirmée] Les deux jeux de données présentent un niveau élevé de chevauchement spatial et partagent le même intervalle de valeurs.")
        print("Cela démontre que, sous fatigue contrôlée ou anomalie cardiaque aiguë, les indicateurs cardiaques (HRV) partagent un même repère physiologique.")
    else:
        print("[Chevauchement faible] Les distributions de HRV sont relativement distinctes.")

if __name__ == "__main__":
    analyze_hrv_correlation()