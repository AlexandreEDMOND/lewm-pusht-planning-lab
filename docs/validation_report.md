# Rapport de validation

Ce document regroupe les vérifications techniques, les résultats expérimentaux
publiés et leurs principales limites. Le suivi des travaux appartient à la
[roadmap](../ROADMAP.md).

Date de validation : **31 juillet 2026**.

## Provenance

| Élément | Référence |
| --- | --- |
| Démonstrateur CEM, évaluation | `8bae6ce10f8694179212a0c1de268b3759401738` |
| Démonstrateur CEM, post-traitement (régénération finale) | `9fcac93f9990ebc704863e92c0a628b23e3e12ef` |
| Instrumentation LeWM | `7246b262be75098f880caacaa7abf8f6c55de22b` |
| Évaluation on-policy, dépôt principal | `f6b74e7b6f235c1b683bd1af590186aca79626ac` |
| Post-traitement on-policy | `da86f77e61752588d764ccf1121d9c2e85af79dc` |
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
| Tests unitaires LeWM | 35/35 | Les contrats temporels, schémas, décodeurs, trace compacte et invariants CEM testés sont respectés |
| Démonstration depuis un clone propre | Succès | `check_phase0`, pytest et la commande de démonstration passent depuis un clone neuf |
| Épisodes de la démo | 2/2 (3876/16, 1766/2) | Résultats observés : succès puis échec, conformes aux cas connus |
| Décisions par épisode | 2 (offsets 0 et 25) | Vérifié depuis `decision_index_per_action` de l'exécution brute |
| Itérations, population, élites | 30, 300, 30 | Vérifié depuis les traces compactes |
| Correspondance plan/action | 4/4 exactes | Les plans sélectionnés égalent les actions normalisées exécutées à `2e-6` |
| Valeurs numériques | Aucun NaN/Inf | Les traces compactes (hors bourrage documenté) et le CSV publié sont finis |
| Taille des traces compactes | 4 × ~2,4 Mo, 9,46 Mo au total | Limites 10 Mo/fichier et 20 Mo au total respectées |
| SHA-256 des artefacts | Recalculés | Les hashes du manifeste correspondent aux fichiers publiés |
| Chemins publiés | Portables | Aucun chemin `/home/...` dans les fichiers versionnés |
| Cohérence Git et Markdown | Succès | `git diff --check` propre, liens Markdown valides, provenance propre |
| Déterminisme du post-traitement | Vérifié | Deux reruns produisent des fichiers identiques (voir ci-dessous) |

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

Les 35 tests se répartissent ainsi :

- 2 tests sur la trace CEM, le plan final et la non-régression des actions ;
- 6 tests sur l'alignement on-policy, la fréquence de replanification, les
  branches factuelles, la normalisation et les schémas ;
- 11 tests sur les splits, indices temporels, blocs d'actions, catégories
  physiques et sorties des trois décodeurs ;
- 16 tests sur la trace compacte de la démo : schéma, sélection déterministe
  des candidats, conservation des élites, correspondance plan/actions,
  offsets 0 et 25, alignement actions/observations, rejet d'une couverture
  incomplète, rejet d'une provenance dirty, portabilité, hashes, non-écrasement
  des expériences on-policy et déterminisme du post-traitement.

## Démonstrateur CEM reproductible

La [démonstration](cem_reproducible_demo.md) a été exécutée depuis un clone
propre avec le protocole officiel. Résultats observés : épisode 3876 (départ
16) **réussi**, épisode 1766 (départ 2) **échoué**, conformes aux cas connus
sélectionnés à l'avance.

| Épisode | Départ | Succès | Erreur T finale | Erreur normalisée | Décisions | Offsets | Plan = actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3876 | 16 | Oui | 9,07 px · 3,54° | 0,945 | 2 | 0, 25 | 2/2 |
| 1766 | 2 | Non | 13,26 px · 4,83° | 1,660 | 2 | 0, 25 | 2/2 |

Planification : 10,74 s et 10,54 s par appel MPC batch (deux environnements),
soit ≈ 5,37 s par décision et par épisode (estimation). Le rerun propre a
reproduit les issues ; les plans diffèrent au niveau du bit de l'étude
précédente (réductions GPU non déterministes entre compositions de batch),
sans changement d'issue.

L'avertissement Gymnasium est mesuré par épisode : les composantes de vitesse
(5–6 de l'état) sortent de l'espace déclaré `[0, 512]` dès l'action 0 dans les
deux épisodes (42/51 et 40/51 frames) ; l'épisode réussi y est aussi soumis.
Incohérence de spécification, facteur descriptif, pas cause démontrée.

Deux reruns du seul post-traitement produisent des fichiers identiques :
CSV, JSON (sidecars et manifeste), PNG, GIF et traces compactes ont les mêmes
SHA-256, le manifeste compris (les temps de planification sont relus depuis
l'exécution brute et ne sont pas remesurés).

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

- [Démonstrateur CEM — manifeste](results/cem_demo_manifest.json)
- [Démonstrateur CEM — métriques par épisode](results/cem_demo_episode_metrics.csv)
- [Démonstrateur CEM — traces compactes](results/cem_demo_compact/)
- [Démonstrateur CEM — animations](assets/cem_demo_success.gif) · [échec](assets/cem_demo_failure.gif) · [synthèse](assets/cem_demo_overview.png)
- [Résultats on-policy JSON](results/on_policy_cem_results.json)
- [Métriques on-policy par épisode](results/on_policy_cem_episode_metrics.csv)
- [Métriques on-policy par frame](results/on_policy_cem_frame_metrics.csv)
- [Résultats de généralisation JSON](results/rollout_generalization_results.json)
- [Métriques de généralisation par épisode](results/rollout_generalization_episode_metrics.csv)
- [Métriques de généralisation par frame](results/rollout_generalization_frame_metrics.csv)
