# T4REC XLNET PROJECT - TIMELINE

**Chronologie complète du projet de recommandation bancaire T4Rec XLNet**

---

## DÉFINITIONS IMPORTANTES

### **Colonne vs Feature**

| Terme | Définition | Exemple |
|-------|------------|---------|
| **Colonne** | Champ brut dans votre dataset Dataiku | `MNT_EPARGNE`, `NB_AUTOMOBILE_12DM` |
| **Feature** | Donnée transformée/encodée utilisée par le modèle | `MNT_EPARGNE` → séquence [5,12,8,15,3,7,9,11] |

**Processus de transformation** :
```
Colonne brute → Transformer (SequenceTransformer/CategoricalTransformer) → Feature pour modèle
```

**Exemple concret** :
- **Colonne** : `NBCHQEMIGLISS_M12` = [1250, 890, 1100, 950] (montants réels)
- **Feature** : `NBCHQEMIGLISS_M12_encoded` = [15, 8, 12, 9] (tokens pour T4Rec)

---

## TIMELINE DU PROJET

### **PHASE 0 : SETUP INITIAL**
**Date** : Début du projet  
**Objectif** : Comprendre l'environnement et les contraintes

#### Découvertes
- **Environnement** : Dataiku + T4Rec 23.04.00
- **Dataset source** : `BASE_SCORE_COMPLETE_prepared`
- **Contrainte critique** : Pas de PyTorch pur, doit utiliser T4Rec
- **Problème API** : T4Rec 23.04.00 a des limitations vs documentation en ligne

#### Résultats
- Identification des contraintes techniques
- Développement toolkit custom `t4rec_toolkit`

---

### **PHASE 1 : POC INITIAL - 560 ÉCHANTILLONS**
**Date** : Premier test de faisabilité  
**Données** : 560 lignes (échantillon test)

#### Sélection Colonnes POC

| Type | Colonnes Utilisées | Méthode Sélection | Nombre |
|------|-------------------|-------------------|---------|
| **Séquentielles** | `nbchqemigliss_m12`, `nb_automobile_12dm`, `mntecscrdimm`, `mnt_euro_r_3m`, `nb_contacts_accueil_service` | **Intuition métier** | 5 |
| **Catégorielles** | `dummy:iac_epa:03`, `dummy:iac_epa:01`, `dummy:iac_epa:02` | **Segmentation client** | 3 |
| **Target** | `souscription_produit_1m` | **Objectif business** | 1 |
| **Total colonnes** | - | - | **9** |
| **Total features** | - | Après transformation | **~10** |

**Note** : Noms en minuscules (données test)

#### Configuration Modèle POC
```
Data: 560 lignes, 9 colonnes → 10 features
Architecture: 128D embedding, 2 layers, 4 heads
Training: 10 époques, batch 32
Temps: 2-3 minutes (CPU)
```

#### Résultats POC
- **Accuracy** : 61.61%
- **Status** : **SUCCESS - POC validé**
- **Leçon** : T4Rec fonctionne mais API 23.04.00 instable

---

### **PHASE 2 : ANALYSE VRAIES DONNÉES - 1000 ÉCHANTILLONS**
**Date** : Passage aux données Dataiku réelles  
**Données** : 1000 lignes (échantillon rapide)

#### Découvertes Critiques
- **Dataset réel** : `BASE_SCORE_COMPLETE_prepared`
- **Colonnes totales** : **541 colonnes** (énorme dataset)
- **Format colonnes** : **MAJUSCULES** (vs minuscules du test)
- **Qualité** : Excellente (0% nulls sur échantillon)

#### Problème Noms de Colonnes

| POC (Test) | Données Réelles | Status |
|------------|-----------------|--------|
| `nbchqemigliss_m12` | `NBCHQEMIGLISS_M12` | ✅ Trouvé |
| `nb_automobile_12dm` | `NB_AUTOMOBILE_12DM` | ✅ Trouvé |
| `mnt_euro_r_3m` | `MNT_EURO_R` | ✅ Similaire |
| `dummy:iac_epa:03` | `IAC_EPA` | ✅ Simplifié |
| `souscription_produit_1m` | `SOUSCRIPTION_PRODUIT_1M` | ✅ Trouvé |

#### Analyse Target Réelle
- **Classes uniques** : 135 produits
- **Distribution** : Déséquilibrée (majorité "Aucune_Proposition")
- **Challenge** : Classe majoritaire dominante

#### Résultats Phase 2
- **Compatibilité** : 100% des colonnes trouvées (avec adaptation noms)
- **Volumétrie estimée 2024** : ~100,000 lignes
- **Mémoire estimée** : ~40GB pour dataset complet
- **Conclusion** : Faisable avec sélection intelligente

---

### **PHASE 3 : SÉLECTION MÉTIER - 12 COLONNES**
**Date** : Optimisation pour 10K lignes  
**Données** : 10,000 lignes cibles

#### Méthode de Sélection

**Critères de sélection** :
1. **Pertinence métier bancaire** (40%)
2. **Qualité des données** (30%) 
3. **Diversité des aspects** (20%)
4. **Performance technique** (10%)

#### Colonnes Sélectionnées Final

| Catégorie | Colonnes | Rationale Métier | Features Générées |
|-----------|----------|------------------|-------------------|
| **Comportement Épargne** | `MNT_EPARGNE`, `NB_EPARGNE`, `TAUX_SATURATION_LIVRET` | Capacité et appétit épargne | 3 features séquentielles |
| **Sophistication** | `MNT_EP_HORS_BILAN`, `NB_EP_BILAN` | Niveau sophistication client | 2 features séquentielles |
| **Activité Transactionnelle** | `NBCHQEMIGLISS_M12`, `MNT_EURO_R` | Fréquence et volume usage | 2 features séquentielles |
| **Segmentation** | `IAC_EPA`, `TOP_EPARGNE`, `TOP_LIVRET` | Profil et segment client | 3 features catégorielles |
| **Engagement** | `NB_CONTACTS_ACCUEIL_SERVICE`, `NB_AUTOMOBILE_12DM` | Relation et fidélité | 2 features catégorielles |

**Total** : **12 colonnes** → **12 features** après transformation

#### Pourquoi 12 et pas 541 ?

| Problème avec 541 | Solution avec 12 |
|-------------------|------------------|
| Mémoire explosive (20GB+) | Mémoire gérable (500MB) |
| Temps entraînement (jours) | Temps raisonnable (30min) |
| Overfitting garanti | Apprentissage stable |
| Complexité ingérable | Architecture optimisée |

---

### **PHASE 4 : PIPELINE PRODUCTION - 10K LIGNES**
**Date** : Pipeline optimisé pour production  
**Données** : 10,000 lignes

#### Configuration Production

```
Données:
- Source: BASE_SCORE_COMPLETE_prepared
- Échantillon: 10,000 lignes
- Colonnes sélectionnées: 12 (sur 541 disponibles)
- Processing: Chunks de 2,000 lignes

Architecture:
- Embedding: 128D
- Layers: 2
- Heads: 4
- Sequence Length: 12
- Batch Size: 32
- Epochs: 15

Features:
- Séquentielles: 6 colonnes → 6 features
- Catégorielles: 6 colonnes → 6 features
- Total: 12 features pour modèle
```

#### Innovations Pipeline

1. **Auto-correction colonnes** : Trouve automatiquement noms MAJUSCULES
2. **Processing chunks** : Gestion mémoire optimisée
3. **Hybrid approach** : T4Rec preprocessing + PyTorch model
4. **Fallback strategies** : Robuste aux erreurs API T4Rec

#### Résultats Estimés Production
- **Accuracy attendue** : 65-75%
- **Temps entraînement** : 20-30 minutes
- **Paramètres modèle** : ~450K
- **Mémoire utilisée** : ~500MB

---

## ÉVOLUTION DES DONNÉES

### **Taille des Échantillons**

| Phase | Lignes | Colonnes Utilisées | Features Générées | Temps Training | Accuracy |
|-------|--------|-------------------|-------------------|----------------|----------|
| **POC** | 560 | 9 | 10 | 2-3 min | 61.61% |
| **Analyse** | 1,000 | - | - | - | - |
| **Production** | 10,000 | 12 | 12 | 20-30 min | 65-75% (estimé) |
| **Future GPU** | 100,000+ | 30-50 | 30-50+ | 2-4 heures | 75-85% (cible) |

### **Évolution Sélection Colonnes**

#### POC (Intuition)
```
Choix: Intuition métier simple
Méthode: "Ces colonnes semblent importantes"
Résultat: 9 colonnes, fonctionne mais basique
```

#### Production (Métier + Technique)
```
Choix: Analyse métier + contraintes techniques
Méthode: Pertinence bancaire + qualité + performance
Résultat: 12 colonnes optimales, équilibre parfait
```

#### Future GPU (Statistique + ML)
```
Choix: Analyse corrélation + mutual information + feature engineering
Méthode: Sélection automatique basée sur prédictivité
Résultat: 30-50 colonnes + features engineered
```

---

## TRANSFORMATION COLONNES → FEATURES

### **Exemple Concret**

#### Colonne Séquentielle
```
Colonne brute: MNT_EPARGNE
Valeurs: [15000, 12000, 18000, 16000, 14000, 17000, 13000, 19000, 15500, 16200]

↓ SequenceTransformer ↓

Feature encodée: [15, 12, 18, 16, 14, 17, 13, 19, 15, 16]
(Vocabulaire: 0-99, normalisation par quantiles)
```

#### Colonne Catégorielle
```
Colonne brute: IAC_EPA
Valeurs: ["PREMIUM", "STANDARD", "PREMIUM", "CLASSIC"]

↓ CategoricalTransformer ↓

Feature encodée: [2, 1, 2, 0]
(Mapping: CLASSIC=0, STANDARD=1, PREMIUM=2)
```

### **Processus Complet**
```
541 colonnes disponibles
    ↓
12 colonnes sélectionnées (métier)
    ↓
Transformation via toolkit
    ↓
12 features pour T4Rec
    ↓
Modèle XLNet
    ↓
Prédictions produits bancaires
```

---

## STRATÉGIE FUTURE AVEC GPU

### **Phase GPU 1 : Extension (30 colonnes)**
- **Méthode** : Corrélation + Mutual Information automatique
- **Data** : 50,000 lignes
- **Architecture** : 256D, 4 layers, 8 heads
- **Temps** : 1-2 heures

### **Phase GPU 2 : Full Scale (50+ colonnes)**
- **Méthode** : Feature engineering avancé + interactions
- **Data** : 200,000+ lignes (dataset complet)
- **Architecture** : 512D, 6 layers, 16 heads
- **Temps** : 4-8 heures

### **Phase GPU 3 : Optimisation**
- **Méthode** : Hyperparameter tuning + ensemble
- **Data** : Multi-années
- **Architecture** : Variable selon tuning
- **Temps** : Jours (avec AutoML)

---

## RÉSUMÉ ACTUEL

**Où on en est** :
- ✅ POC validé (61.61% accuracy)
- ✅ Vraies données analysées (541 colonnes disponibles)
- ✅ Sélection métier optimisée (12 colonnes)
- ✅ Pipeline production prêt (10K lignes)
- 🚧 Test production en cours

**Prochaine étape** :
- Lancer pipeline 10K lignes
- Analyser résultats
- Préparer extension GPU avec sélection automatique

---

**Date de dernière mise à jour** : 2024-08-07 