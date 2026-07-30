# Rapport de validation

Ce document regroupe les vérifications techniques, les résultats expérimentaux
publiés et leurs principales limites. Le suivi des travaux appartient à la
[roadmap](../ROADMAP.md).

Date de validation : **30 juillet 2026**.

## Provenance

| Élément | Référence |
| --- | --- |
| Évaluation on-policy, dépôt principal | `f6b74e7b6f235c1b683bd1af590186aca79626ac` |
| Post-traitement on-policy | `da86f77e61752588d764ccf1121d9c2e85af79dc` |
| Instrumentation LeWM | `aac1036b9eb35e7499d381964b1213eb277bb132` |
| Checkpoint officiel | `a7f1ae0cfbfad8aca613f737d66d12220fa2a8e345c5b46de8b89496c44ced62` |
| Seed CEM | `42` |
| Matériel validé | NVIDIA RTX 3090, 24 Go |
| Environnement | Python 3.10.20, PyTorch 2.13.0, CUDA 13.0 |

Les chemins propres à la machine ne sont pas enregistrés dans les résultats
versionnés. Les fichiers lourds restent sous `$STABLEWM_HOME`.

## Vérifications techniques

| Vérification | Résultat | Interprétation |
| --- | ---: | --- |
| Python, CUDA, GPU, dataset et checkpoints | Succès | Les dépendances matérielles et les assets attendus sont disponibles |
| Tests unitaires LeWM | 19/19 | Les contrats temporels, schémas, décodeurs et invariants CEM testés sont respectés |
| Post-traitement depuis les artefacts bruts | Succès | Les rapports peuvent être reconstruits sans rejouer la simulation |
| Couverture on-policy | 24/24 dans chaque condition | Les deux contrôleurs utilisent exactement les mêmes couples épisode/départ |
| Correspondance plan/action | 48/48 exactes | Les métriques portent sur les actions réellement sélectionnées et envoyées |
| Valeurs numériques | Aucun NaN/Inf | Les CSV publiés ne contiennent pas de mesure non finie |
| Cohérence Git et Markdown | Succès | Les patches ne contiennent pas d'erreur d'espace blanc et les médias liés existent |

Commande du contrôle matériel :

```bash
bash scripts/check_phase0.sh --require-cuda --require-assets
```

Commande des tests :

```bash
env PYTHONPATH=third_party/le-wm \
  uv run --project . --with pytest \
  pytest -q third_party/le-wm/tests
```

Les 19 tests se répartissent ainsi :

- 2 tests sur la trace CEM, le plan final et la non-régression des actions ;
- 6 tests sur l'alignement on-policy, la fréquence de replanification, les
  branches factuelles, la normalisation et les schémas ;
- 11 tests sur les splits, indices temporels, blocs d'actions, catégories
  physiques et sorties des trois décodeurs.

## Décodage du latent

Les décodeurs utilisent le latent global de 192 dimensions du checkpoint
officiel, sans modifier LeWM ni son coût de contrôle.

| Mesure sur 2 048 images de test | Résultat |
| --- | ---: |
| PSNR du décodeur convolutionnel | 26,67 dB |
| SSIM | 0,924 |
| IoU du premier plan | 0,787 |
| Erreur médiane du pousseur, décodeur structuré | 5,97 px |
| Erreur médiane du T | 4,21 px |
| Erreur angulaire médiane du T | 0,89° |

**Interprétation.** Le latent conserve assez d'information pour reconstruire la
configuration globale de PushT et mesurer une erreur physique. Le décodeur
structuré connaît toutefois la géométrie du simulateur : il sert d'instrument de
diagnostic, pas de preuve autonome de qualité visuelle.

Détails : [faisabilité du décodage visuel](visual_decoder_feasibility.md).

## Généralisation des rollouts

L'évaluation utilise une fenêtre uniforme dans chacun des 128 épisodes de test,
indépendamment des sorties du modèle.

| Mesure terminale à `t=35` | Médiane | P95 |
| --- | ---: | ---: |
| MSE latente | 0,0374 | 0,2225 |
| Erreur de position du T | 6,74 px | 25,35 px |
| Erreur angulaire du T | 2,02° | 18,30° |

La MSE latente est modérément corrélée à l'erreur physique sous les mêmes actions
expertes (`ρ≈0,33–0,35`). Son AUC pour prédire les échecs du contrôleur qui
choisit d'autres actions n'est que de `0,57`.

**Interprétation.** Le comportement médian est précis, mais la queue d'erreur est
importante. Une erreur mesurée sur une trajectoire experte ne suffit pas à
estimer la fiabilité du contrôle.

Détails : [généralisation des rollouts](rollout_generalization.md).

## Contrôle et erreur on-policy

Les deux conditions utilisent le checkpoint officiel et les mêmes 24 cas
stratifiés. Cet ensemble contient volontairement des cas difficiles : ses ratios
de succès ne sont pas des estimations de performance dans la population.

| Condition | Succès | Décisions/épisode | Rollouts CEM/épisode | Temps/épisode |
| --- | ---: | ---: | ---: | ---: |
| `receding_horizon=5` | 21/24 | 2 | 18 000 | 7,97 s |
| `receding_horizon=1` | 7/24 | 10 | 90 000 | 35,75 s |

Comparaison appariée :

- 7 succès communs ;
- 14 succès uniquement avec `receding_horizon=5` ;
- aucun succès uniquement avec `receding_horizon=1` ;
- 3 échecs communs.

Le second réglage utilise exactement cinq fois plus de rollouts et 4,49 fois
plus de temps mur observé.

À 25 actions avec `receding_horizon=5`, l'erreur médiane du T est de 9,23 px
chez les succès et 23,26 px chez les trois échecs, soit une AUC descriptive de
1,00. Avec `receding_horizon=1`, l'AUC est de 0,55 pour l'erreur factuelle à
cinq actions et de 0,45 pour le flux exécuté à 25 actions.

**Interprétation.** L'erreur longue portée est informative pour les trois échecs
observés avec le protocole officiel. Elle n'explique pas la forte dégradation
provoquée par une replanification plus fréquente. Replanifier davantage n'est
donc pas automatiquement bénéfique lorsque le modèle, le coût et le bloc
d'actions restent inchangés.

Détails : [erreur on-policy sous CEM](on_policy_cem_error.md).

## Erreurs et limites connues

### Bornes déclarées par PushT

Gymnasium signale que certaines observations sortent de leur espace déclaré.
Les positions XY du pousseur peuvent atteindre environ `-800` à `901`, et ses
vitesses peuvent être négatives alors que l'espace annonce `[0,512]`.

- `receding_horizon=5` sort des bornes dans 6/24 épisodes, dont 2 échecs ;
- `receding_horizon=1` sort des bornes dans 9/24 épisodes, dont 8 échecs.

Les observations ont été conservées sans correction ni masquage. Cette
incohérence peut contribuer à certains échecs, mais ne les explique pas tous.

### Portée statistique

- Les 24 cas de contrôle sont stratifiés par risque et non tirés comme un
  échantillon représentatif.
- L'AUC de 1,00 repose sur trois échecs seulement ; elle est descriptive et ne
  démontre aucun lien causal.
- Les erreurs physiques dépendent d'un décodeur imparfait. Les rapports publient
  également son plafond encode→decode pour séparer autant que possible les deux
  sources d'erreur.

## Artefacts vérifiables

- [Résultats on-policy JSON](results/on_policy_cem_results.json)
- [Métriques on-policy par épisode](results/on_policy_cem_episode_metrics.csv)
- [Métriques on-policy par frame](results/on_policy_cem_frame_metrics.csv)
- [Résultats de généralisation JSON](results/rollout_generalization_results.json)
- [Métriques de généralisation par épisode](results/rollout_generalization_episode_metrics.csv)
- [Métriques de généralisation par frame](results/rollout_generalization_frame_metrics.csv)
