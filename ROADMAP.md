# Roadmap

Cette roadmap sépare le **démonstrateur pédagogique actuel** des travaux de
**recherche**. Le README ne porte que le premier : il explique LeWM, montre le
rollout visuel et rend la recherche CEM observable. Les études plus longues,
leurs limites et les comparaisons futures vivent dans [research/](research/).

Une fonctionnalité locale n'est donc pas automatiquement considérée comme
terminée : elle doit aussi avoir une preuve versionnée et lisible.

Dernière validation : **31 juillet 2026**. Résultats et vérifications :
[research/validation_report.md](research/validation_report.md). Le
[démonstrateur CEM reproductible](research/cem_reproducible_demo.md) relie la
recherche CEM, les plans, les actions exécutées, les futurs prédits, la
trajectoire réelle et le résultat final sur deux épisodes fixes.
L'[audit initial](research/project_audit_2026-07-30.md) reste disponible comme
photographie historique du dépôt avant l'étude on-policy.

## Objectif actuel : démonstrateur visuel LeWM + CEM

| Élément | État | Ce qui est montré |
| --- | --- | --- |
| Chaîne world model | Terminé | Image → encodeur → embedding + action → predictor → embedding suivant → décodeur visuel, en boucle. |
| Rollout visuel | Terminé, avec une convention à connaître | Quatre GIFs comparent le réel et le rollout de **18 pas du modèle** ; un pas représente actuellement un bloc de 5 actions, donc 90 actions PushT. |
| CEM observable | Terminé | L'image de départ et l'objectif guident 300 trajectoires candidates ; les 30 élites, le meilleur plan et leur convergence sont visibles. |
| Bench de convergence CEM | Terminé | 60 itérations × 300 trajectoires : 95 % du gain latent observé à 9 900 trajectoires et 4,63 s GPU synchronisées ; 8,22 s au total. |
| Comparaison VLA | Hors objectif actuel | Seul le protocole est préparé dans la recherche ; aucun résultat VLA n'est revendiqué. |

Le démonstrateur conserve volontairement 18 **pas du modèle** (90 actions
PushT), car ils rendent visible la dérive d'un rollout long tout en respectant
la granularité native de LeWM, cinq actions par pas. Il conserve aussi 300
candidats et 30 élites : cette population suffit à visualiser CEM et le bench
mesure désormais son coût réel. Passer à 18 actions élémentaires ou 500
candidats n'apporterait pas de réponse supplémentaire à l'objectif actuel.

## Travaux de recherche (hors objectif actuel)

| Jalon | État | Preuve actuelle | Prochaine sortie |
| --- | --- | --- | --- |
| Phase 0 — bootstrap | À auditer proprement | Contrôle complet réussi sur la machine actuelle | Installation depuis un clone neuf, hashes et manifeste d'environnement |
| Phase 1 — référence CEM | Reproductible sur cas fixes | Démo end-to-end (épisodes 3876/16 et 1766/2) depuis un clone propre | Évaluation représentative |
| Phase 2 — instrumentation | Terminée | Traces compactes versionnées, provenance propre, hashes et schéma v1 | Temps par itération individuelle |
| Phase 3 — visualisation | Terminée pour la démo | GIFs de trajectoire complète exécutée, plan, coût et prédit/réel publiés | Animation de toute évaluation future |
| Phase 3 bis — décodeurs | Validée | Rapport, figures et GIFs versionnés | Maintenir ; ne plus optimiser sans besoin de contrôle |
| Généralisation offline | Validée | 128 épisodes, CSV/JSON et cas extrêmes versionnés | Aucun blocage pour ce sous-jalon |
| Erreur on-policy | Validée, stratifiée | RH=5/RH=1, artefacts et GIFs | Examiner coût latent/solveur |
| Démo CEM reproductible | Validée | Deux épisodes, manifeste, animations, 35 tests, clone propre | Aucun blocage pour ce jalon |
| Phase 4 — entraînement local | Smoke test seulement | Deux batches locaux et configuration 100 époques | Run complet avec et sans SIGReg |
| Phase 5 — baselines | Non commencée | Protocole prévu | CEM vs random shooting à budget égal |
| Phase 6 — probes | Partielle | Décodeur d'état non linéaire | Probes linéaires et comparaison de checkpoints |
| Phase 7 — OOD | Non commencée | Intention | Protocoles visuel et physique |
| Phase 8 — VLA | Non commencée, hors objectif actuel | Protocole équitable versionné | Fine-tuning puis évaluation sur les mêmes épisodes que LeWM+CEM |

Les cases ci-dessous signifient :

- `[x]` : implémenté et vérifié par une validation ou un artefact versionné ;
- `[ ]` : absent, partiel, local seulement ou encore à reproduire proprement.

## Validation de généralisation — terminée

Le [rapport du 25 juillet 2026](research/rollout_generalization.md) évalue une
fenêtre uniforme sur chacun des 128 épisodes de test, sépare mouvement libre,
contact et poussée effective, puis rejoue CEM sur 24 cas stratifiés par risque.

- [x] Évaluer les rollouts `t=35` sur les 128 épisodes de test sans sélection
  selon les sorties du modèle.
- [x] Rapporter médiane, P90, P95 et pires cas.
- [x] Séparer mouvement libre, contact sans mouvement et poussée effective avec
  la géométrie Pymunk réelle.
- [x] Produire les courbes par horizon, la calibration latente/physique et les
  GIFs meilleur, médian et pire.
- [x] Rejouer le CEM officiel sur 24 épisodes couvrant tout le spectre de risque.
- [x] Relier erreurs offline, coût CEM, réussite et meilleure proximité physique
  au but.

**Résultat :** à `t=35`, la médiane est de 6,74 px et 2,02° sur le T, mais le P95
atteint 25,35 px et 18,30°. La MSE latente est modérément corrélée à l'erreur
physique sous les mêmes actions expertes (`ρ≈0,33–0,35`). Elle ne prédit pas les
échecs CEM sous d'autres actions (`AUC=0,57`, 21 succès sur 24 cas stratifiés).

## Validation terminée — erreur on-policy de CEM

1. Sauvegarder, pour chaque décision CEM, les 25 actions choisies et les latents
   prédits correspondants.
2. Capturer les images et états réellement obtenus après chaque bloc de cinq
   actions.
3. Mesurer l'écart latent et physique on-policy aux temps `5,10,…,25`.
4. Comparer `receding_horizon=5` et `receding_horizon=1` sur les mêmes épisodes,
   en rapportant séparément performance brute et coût de calcul.
5. Décider à partir de ce test entre modèle temporel plus fin (`action_block=1`)
   et amélioration du coût/solveur CEM.

**Résultat :** [l'étude on-policy](research/on_policy_cem_error.md) observe une
dérive à 25 actions; RH=1 ne réduit pas l'erreur à cinq actions et obtient
7/24 contre 21/24 pour RH=5 sur les mêmes cas stratifiés. Ces proportions ne
sont pas des taux de réussite populationnels.

## Chemin critique de recherche recommandé

L'ordre suivant permet de prendre une décision à chaque jalon et évite
d'accumuler des expériences décoratives :

1. **Fermer la reproductibilité des phases 0–3** : clone propre, seconde
   exécution, résultats et petite trace publiés, vidéo reliée à la trajectoire.
2. **Établir la référence de contrôle** : utiliser un ensemble représentatif fixé
   avant l'exécution et publier un intervalle de confiance.
3. **Entraîner localement** : run complet SIGReg, ablation sans SIGReg et
   comparaison au checkpoint officiel.
4. **Attribuer le gain au planning** : CEM contre random shooting au même budget,
   puis seulement les balayages de population/horizon qui répondent à une
   question.
5. **Examiner le coût et le solveur** : exploiter le diagnostic on-policy avant
   de tester une nouvelle fréquence temporelle ou `action_block=1`.
6. **Préparer la version 1** : artefacts, limites, licence, citation, release et
   README final.

Les probes, l'OOD, iCEM/MPPI et le VLA viennent après ce chemin critique.

## Définition de terminé

### Niveau A — démonstrateur pédagogique reproductible

- [x] Un clone neuf passe l'installation, le contrôle phase 0 et les 35 tests.
- [x] Une commande courte reproduit une évaluation CEM et son résultat structuré.
- [x] Une métrique de référence, une trace compacte et une animation de planning
  sont téléchargeables avec leurs hashes.
- [x] L'animation relie la recherche CEM à toute la trajectoire exécutée.
- [x] Le README présente PushT, LeWM, CEM et les visualisations avec une
  animation intégrée.
- [x] Montrer et mesurer une recherche CEM suffisamment longue : nombre de
  candidats jusqu'à 95 % du gain latent observé et temps GPU synchronisé.
- [ ] Le dépôt racine possède une licence ; code, dataset et checkpoints ont une
  provenance explicite.

### Niveau B — programme de recherche / version 1 scientifique

Le niveau B correspond à la promesse actuelle « apprendre un world model puis
l'utiliser pour planifier ». Il exige le niveau A, plus :

- [x] Mesure de l'erreur on-policy et comparaison `receding_horizon=5` contre
  `receding_horizon=1`.
- [ ] Évaluation du contrôle sur le protocole représentatif préannoncé, avec
  succès par épisode, dispersion et intervalle de confiance.
- [ ] Entraînement local complet avec SIGReg, checkpoint et courbes.
- [ ] Ablation sans SIGReg et diagnostic du collapse par plusieurs indicateurs.
- [ ] Comparaison checkpoint officiel / checkpoint local sur les mêmes épisodes.
- [ ] Comparaison CEM / random shooting à budget total de rollouts égal.
- [ ] Comparaison LeWM+CEM / VLA à informations visuelles, objectifs et budget
  d'actions égaux ; ne conclure qu'après une évaluation publiée. Ce point est
  explicitement hors du démonstrateur actuel.
- [ ] Balayages annoncés exécutés, ou périmètre réduit explicitement avant
  l'analyse.
- [ ] Release regroupant résultats bruts, configurations, hashes, figures,
  animations et tableau des limites.

Les phases 6 à 8 ne bloquent pas cette version si elles sont présentées comme
extensions. Si elles restent dans le périmètre scientifique central, leurs
critères de sortie redeviennent obligatoires.

## Décisions de cadrage

- **Tâche initiale** : PushT visuel et actions continues 2D, sans robot réel.
- **Contrôle de référence** : protocole officiel LeWM avec horizon `5`, blocs de `5` actions et `receding_horizon=5` ; les `25` actions élémentaires du plan sont exécutées avant replanification.
- **Contrôle en ablation** : mesurer séparément `receding_horizon=1`, puis éventuellement `action_block=1`, afin de quantifier le gain d'une boucle plus fermée sans confondre ce résultat avec la reproduction officielle.
- **Coût principal** : distance entre le latent prédit à l'horizon et le latent de l'image objectif.
- **Référence** : checkpoint LeWM officiel pour la première démo ; entraînement local après validation de la chaîne complète.
- **Matériel cible** : une RTX 3090 de 24 Go. Les expérimentations commencent en précision mixte et avec un batch compatible avec la VRAM mesurée.
- **Comparaison équitable** : les solveurs reçoivent le même budget total de rollouts de modèle.
- **Décodeur visuel** : outil de diagnostic et de communication, pas composant du coût CEM tant que son utilité pour le contrôle n'est pas démontrée.
- **Comparaison VLA** : extension expérimentale. Elle compare le contrôle obtenu à données et épisodes PushT identiques, sans prétendre isoler l'effet du préentraînement ou du langage.

## Détail des phases de recherche

Les phases suivantes développent ou évaluent scientifiquement le système. Elles
ne sont pas nécessaires pour lire ou présenter le démonstrateur, mais leurs
rapports et données sont conservés dans [research/](research/).

## Phase 0 — Bootstrap reproductible

**But :** installer et figer un environnement capable de lancer PushT et l'évaluation officielle.

**État : fonctionnel sur la machine auditée ; installation propre encore à
valider.**

- [x] Créer le projet Python avec `uv` et verrouiller les dépendances.
- [x] Épingler le fork LeWM par sous-module et `stable-worldmodel` dans le lock.
- [x] Ajouter une configuration locale non versionnée pour les chemins de données et de checkpoints.
- [x] Télécharger les assets et contrôler présence, chargement et schéma HDF5.
- [x] Enregistrer une commande unique pour une évaluation déterministe (seed fixée).
- [ ] Enregistrer le driver GPU et les hashes attendus du dataset/checkpoint.
- [ ] Rejouer toute la procédure depuis un clone neuf sur une machine vide.

**Sortie vérifiable :** une commande documentée charge le checkpoint, exécute au moins un épisode PushT et écrit un résultat structuré (seed, configuration, métriques et version du code).

## Phase 1 — Référence MPC avec CEM

**But :** obtenir le contrôle de référence avant toute modification du modèle ou du solveur.

**État : implémenté et exécuté sur cinq épisodes ; résultat fonctionnel mais
insuffisant pour estimer la performance générale.**

- [x] Exécuter CEM avec le coût latent et le checkpoint LeWM.
- [x] Fixer une première configuration : horizon, nombre d'itérations, population, fraction d'élites et bornes d'actions.
- [x] Évaluer sur un petit ensemble fixe d'épisodes/seeds.
- [x] Produire localement une vidéo de chaque épisode contrôlé.
- [x] Mesurer taux de réussite, coût final et temps de planning par pas.
- [ ] Refaire l'exécution depuis un arbre Git propre et comparer aux tolérances annoncées.
- [ ] Versionner ou publier les métriques et une vidéo représentative.
- [ ] Évaluer un ensemble représentatif avec intervalle de confiance.

**Sortie vérifiable :** les mêmes seeds génèrent les mêmes métriques à une tolérance numérique documentée, et les épisodes sauvegardés montrent une politique qui tente effectivement de placer le T dans la cible.

## Phase 2 — CEM instrumenté

**But :** faire du CEM un objet d'étude plutôt qu'un appel opaque au solveur existant.

**État : format de trace, latents, élites, tests unitaires et trace compacte
publiée avec provenance propre ; le temps individuel de chaque itération
reste à enregistrer.**

- [x] Définir un format de trace par décision MPC et itération CEM.
- [x] Enregistrer les séquences candidates, coûts, indices des élites, moyenne et écart-type.
- [ ] Enregistrer le temps individuel de chaque itération.
- [x] Enregistrer les latents des rollouts et le latent objectif nécessaires à l'analyse.
- [x] Vérifier que l'instrumentation ne change pas les actions ni le résultat du CEM de référence.
- [x] Tester la sélection des élites, la mise à jour moyenne/écart-type et les bornes des actions retournées.
- [x] Publier une trace compacte avec provenance propre, hash et description du schéma (schéma v1, research/results/cem_demo_compact/).

**Sortie vérifiable :** pour une décision donnée, une trace permet de reconstruire l'évolution de `μ`, `σ`, les élites et le meilleur coût à chaque itération.

## Phase 3 — Visualisation de planning

**But :** produire une visualisation claire, exportable et fidèle aux traces CEM.

**État : générateur et vidéos disponibles localement. Les GIFs publiés
affichent toute la trajectoire exécutée, les replanifications aux actions 0
et 25, le plan sélectionné, la convergence du coût et le prédit/réel
factuel.**

- [x] Afficher l'environnement réel et toute la trajectoire exécutée.
- [x] Afficher la frame exacte à laquelle la décision tracée est prise.
- [x] Afficher la population d'actions et distinguer candidats, élites et moyenne.
- [x] Afficher les coûts et la contraction de la dispersion au fil des itérations.
- [x] Afficher les rollouts latents sans prétendre qu'une projection 2D est une preuve physique.
- [x] Exporter une vidéo et son métadonnée JSON à partir d'une trace sauvegardée.
- [x] Publier une animation légère, légendée et reliée aux métriques sources (GIFs de la démo, convention de 57 frames documentée dans research/cem_reproducible_demo.md).

**Sortie vérifiable :** une vidéo montre, pour un épisode, la population initialement dispersée, la sélection des élites et la concentration de la distribution jusqu'à l'action exécutée. Les chiffres affichés correspondent aux traces brutes.

## Phase 3 bis — Décodeur visuel et rollouts réel/prédit

**Difficulté estimée : moyenne pour un prototype, élevée pour obtenir des images nettes et physiquement fidèles.**

Le LeWM actuel utilise un unique embedding global issu du token CLS. Il ne conserve pas explicitement une grille spatiale de patches ; un petit décodeur peut donc reconstruire une image, mais la netteté et la géométrie ne sont pas garanties. Le décodeur doit d'abord être validé sur des latents d'images réelles avant d'être utilisé pour juger les rollouts autorégressifs.

**Résultat du test de faisabilité : positif avec réserves.** Le [rapport du 25 juillet 2026](research/visual_decoder_feasibility.md) mesure le décodeur pixel direct à 26,67 dB de PSNR, 0,924 de SSIM et 0,787 d'IoU du premier plan sur 2 048 images de 128 épisodes de test. Les poses sont reconnaissables mais les contours restent flous. Un décodeur structuré `latent -> état PushT -> rendu simulateur` donne des formes exactes avec une erreur moyenne de 5,98 px sur la position du T et 2,24° sur son angle.

Le Transformer inspiré de l'annexe D est moins fidèle avec le même budget de 10 000 images : 22,14 dB de PSNR et 0,663 d'IoU. Sur le protocole officiel, le décodeur convolutionnel appliqué aux latents prédits atteint 27,26 dB, 0,923 de SSIM et 0,802 d'IoU en moyenne sur les frames prédites. À `t=90`, le stress test reste lisible (25,71 dB et 0,768 d'IoU), mais le diagnostic physique révèle 13,42 px d'erreur moyenne sur le T et 7,49° sur son orientation.

Le **décodeur pixel depuis le latent CLS reste la preuve visuelle principale**, car il montre directement ce que le world model conserve et prédit sans utiliser d'état privilégié. Le décodeur structuré est un diagnostic secondaire : il rend les erreurs de pose très lisibles, mais exploite les annotations physiques et la géométrie connue de PushT.

**But :** transformer les latents en images uniquement pour observer où et quand la prédiction s'écarte du vrai rollout PushT.

- [x] Figer l'encodeur et le prédicteur, puis entraîner un décodeur léger `D(z_t) -> o_t` sur les images PushT.
- [x] Entraîner et évaluer un décodeur structuré vers la pose PushT, puis rendre cette pose avec le simulateur pour obtenir des formes nettes.
- [x] Implémenter le décodeur Transformer de l'annexe D du papier : projection du CLS, `196` requêtes apprises, cross-attention et projection vers des patches RGB `16 × 16`.
- [x] Comparer Transformer et convolution sur le même split, les mêmes latents et le même budget d'images avant tout changement d'échelle.
- [x] Mesurer séparément la reconstruction de latents réels encodés, qui teste le décodeur, et la reconstruction de latents prédits, qui teste toute la chaîne.
- [x] Rapporter perte pixel, PSNR, SSIM, IoU du premier plan et erreurs physiques sur des épisodes tenus hors entraînement, en plus d'une inspection visuelle. LPIPS est écarté car moins interprétable que les erreurs de pose sur PushT.
- [x] Ajouter un test de faisabilité court sur la RTX 3090 avant l'entraînement complet. Arrêter ou revoir la représentation si le décodeur ne restitue pas au minimum la pose du T, la cible et le pousseur à partir d'un latent réel.
- [x] Définir sans ambiguïté la résolution temporelle du benchmark de 18 images.
  - Le modèle actuel utilise un horizon de `5` blocs contenant chacun `5` actions 2D, soit `25` actions élémentaires, et ne produit qu'un latent par bloc.
  - Pour obtenir exactement une image prédite après chacune des `18` actions élémentaires, entraîner une variante `action_block=1` avec un horizon d'évaluation de `18`.
  - Conserver en baseline moins coûteuse un rollout de `18` pas du modèle actuel, correspondant à `90` actions élémentaires, clairement étiqueté comme tel.
- [x] Sélectionner quatre épisodes de test avec des poses initiales différentes, sans les utiliser pour choisir le décodeur ou ses hyperparamètres.
- [x] Reproduire d'abord le protocole officiel : trois images de contexte, blocs de cinq actions, puis affichage aux temps `t ∈ {0,5,10,…,35}`.
- [x] Pour chaque épisode, partir des mêmes observations initiales et appliquer au modèle les actions réellement exécutées dans la démonstration PushT sauvegardée.
- [x] Produire un GIF principal avec : image réelle, reconstruction indépendante du vrai latent et décodage du latent prédit en boucle ouverte.
- [x] Ajouter le rendu structuré et les erreurs de pose comme quatrième panneau diagnostique clairement étiqueté.
- [x] Après la reproduction officielle, produire un stress test de `18` pas du modèle actuel, soit `90` actions élémentaires.
- [x] Produire un panneau récapitulatif des quatre GIFs et les métriques par horizon afin de rendre visible l'accumulation d'erreur.
- [x] Tester que la génération des GIFs est déterministe à partir d'un checkpoint, d'une seed, d'une liste d'actions et d'identifiants d'épisodes sauvegardés (checksums SHA-256 identiques sur deux exécutions CUDA).

**Sortie vérifiable :** quatre GIFs reproductibles comparent réel, reconstruction indépendante et prédiction autorégressive sur des épisodes tenus à l'écart. Le rapport sépare explicitement la limite du décodeur de l'erreur du world model, puis distingue la reproduction officielle jusqu'à `t=35` du stress test de `18` blocs.

## Phase 4 — Entraînement local et stabilité de représentation

**But :** reproduire un entraînement LeWM sur la RTX 3090 et mesurer les effets de SIGReg.

**État : configuration et script prêts. Un smoke test de deux batches a mesuré
environ 12,4 Gio de VRAM, mais il ne valide ni convergence, ni stabilité, ni
contrôle.**

- [ ] Lancer l'entraînement LeWM de référence avec logs de `L_prediction`, `L_SIGReg`, VRAM et débit.
- [ ] Sauvegarder checkpoints, config complète et seed.
- [ ] Mesurer l'erreur de prédiction sur plusieurs horizons et la distribution/variance des embeddings.
- [ ] Relancer une expérience sans régularisation pour caractériser le collapse ou son absence.
- [ ] Implémenter une variante VICReg seulement après avoir validé la variante sans régularisation.

**Sortie vérifiable :** chaque run possède un identifiant, une configuration immuable, des courbes de pertes et une évaluation MPC. L'analyse ne conclut à un collapse qu'à partir de plusieurs indicateurs, pas d'une projection 2D seule.

## Phase 5 — Baselines et ablations de planning

**But :** attribuer les gains au planning plutôt qu'au budget de calcul.

**État : non commencé. La comparaison CEM/random shooting est obligatoire pour
la version 1 ; iCEM et MPPI sont optionnels.**

- [ ] Implémenter ou instrumenter random shooting avec le même coût latent et le même budget total de rollouts que CEM.
- [ ] Évaluer iCEM avec le même protocole ; ajouter MPPI seulement si la comparaison reste lisible.
- [ ] Balayer `N ∈ {32,64,128,256,512}`.
- [ ] Balayer `H ∈ {4,8,12,16}`.
- [ ] Rapporter moyenne, dispersion, temps de planning et taux de réussite sur les mêmes seeds.

**Sortie vérifiable :** chaque figure/tableau annonce explicitement le budget de rollouts, l'horizon, les seeds et le modèle utilisé. CEM et random shooting sont comparables sans ambiguïté.

## Phase 6 — Analyse du latent

**But :** vérifier quantitativement que le latent encode des variables utiles au contrôle.

**État : partiellement amorcé par le décodeur structuré MLP sur le checkpoint officiel. Les probes linéaires, le contact, la distance au but et les comparaisons de checkpoints restent à faire.**

- [ ] Générer ou récupérer les cibles de vérité terrain : pose du T, pose du pousseur, distance au but et contact.
- [ ] Geler l'encodeur et séparer les données train/validation/test par épisode.
- [ ] Entraîner des probes légers linéaires, puis documenter leur métrique appropriée (erreur de position, erreur angulaire, précision contact).
- [ ] Comparer checkpoint officiel, LeWM entraîné localement et variante sans SIGReg.

**Sortie vérifiable :** un tableau de probes hors échantillon est produit avec protocole de séparation explicite ; aucune donnée du test n'est utilisée pour sélectionner le probe.

## Phase 7 — Robustesse hors distribution

**But :** tester si contrôle et représentation résistent à des variations pertinentes.

**État : extension non commencée.**

- [ ] Identifier les facteurs de variation PushT disponibles dans stable-worldmodel.
- [ ] Définir des protocoles séparés pour apparence (textures/couleurs) et dynamique.
- [ ] Évaluer le modèle et le contrôleur sur les mêmes seeds entre conditions nominales et OOD.
- [ ] Rapporter réussite, coût final, erreur multi-step et éventuellement signal de surprise.

**Sortie vérifiable :** les résultats distinguent sans ambiguïté variation visuelle et variation physique, avec configurations versionnées.

## Phase 8 — Baseline Vision-Language-Action

**Statut : extension non bloquante pour la version 1. Difficulté élevée ; ce
n'est pas une comparaison « plug-and-play ».**

PushT fournit une image et des actions continues 2D, mais pas les états articulaires, espaces d'actions robotiques ni annotations linguistiques attendus par les VLA courants. Un VLA préentraîné ne peut donc pas être évalué directement : il faut adapter le dataset, la tête d'action et la boucle d'inférence, puis le fine-tuner.

Cette extension doit vivre dans un environnement séparé. Le projet principal
est figé en Python 3.10, tandis que l'extra LeRobot du
[`stable-worldmodel` actuel](https://github.com/galilai-group/stable-worldmodel)
annonce Python 3.12 ou plus. SmolVLA fournit par ailleurs une référence de
fine-tuning sur A100, pas une garantie de protocole utile sur RTX 3090.

**But :** comparer LeWM+CEM à une politique VLA fine-tunée sur PushT avec un protocole qui rend visibles les avantages, les coûts et les biais de chaque approche.

- [ ] Commencer par un test de capacité mémoire et de latence sur la RTX 3090 avant toute conversion complète du dataset.
- [ ] Utiliser en première option un VLA compact pris en charge par LeRobot, par exemple SmolVLA, plutôt qu'un modèle de plusieurs milliards de paramètres nécessitant une infrastructure distincte.
- [ ] Figer la version du dépôt, le checkpoint amont, la licence, le nombre de paramètres et la provenance des données de préentraînement.
- [ ] Convertir PushT vers le format LeRobot en conservant les séparations par épisode, les images, les états autorisés, les actions 2D, les timestamps et les seeds.
- [ ] Définir une instruction constante et non ambiguë, par exemple « pousser le T dans la cible ». Documenter qu'une instruction constante ne mesure pas la généralisation linguistique du VLA.
- [ ] Adapter et normaliser la sortie du VLA vers les actions continues 2D de PushT ; choisir une longueur de chunk compatible avec la fréquence d'action du protocole LeWM.
- [ ] Interdire les informations privilégiées au VLA si elles ne sont pas accessibles au LeWM lors de la comparaison principale.
- [ ] Fine-tuner avec le même ensemble d'épisodes de démonstration que celui utilisé pour le modèle local, puis sélectionner les hyperparamètres sur une validation séparée.
- [ ] Évaluer VLA et LeWM+CEM sur les mêmes épisodes de test, poses initiales, objectifs, budget d'actions et critère de réussite.
- [ ] Rapporter taux de réussite, coût final, latence par action, VRAM, taille du modèle, quantité de données, temps de fine-tuning et fréquence de contrôle.
- [ ] Ajouter une baseline de politique imitation plus simple, telle qu'ACT ou Diffusion Policy via LeRobot, pour vérifier si le gain éventuel vient réellement du VLA plutôt que d'une politique directe.
- [ ] Présenter séparément :
  - la performance de contrôle ;
  - le coût de calcul et de données ;
  - la capacité du world model à simuler et planifier ;
  - la capacité éventuelle du VLA à exploiter des instructions variées.
- [ ] Si aucun VLA compact ne tient sur 24 Go avec une configuration utile, documenter le test négatif et reporter l'expérience sur un GPU plus grand au lieu de réduire silencieusement le protocole.

**Sortie vérifiable :** un checkpoint VLA adapté à PushT et un rapport reproductible comparent au minimum LeWM+CEM, le VLA et une politique imitation simple sur les mêmes épisodes. Toute différence de préentraînement, d'observation ou de fréquence d'action est explicitement annoncée.

**Références d'implémentation candidates :**

- [LeRobot](https://github.com/huggingface/lerobot), pour le format de données, l'entraînement et l'évaluation unifiés.
- [SmolVLA dans LeRobot](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx), comme premier VLA compact à tester.
- [OpenVLA](https://github.com/openvla/openvla) et [openpi](https://github.com/Physical-Intelligence/openpi), comme variantes ultérieures seulement si l'adaptation et les ressources restent maîtrisées.

## Livrables de la version 1

- [ ] Installation et reproduction validées depuis un clone propre.
- [ ] Checkpoints ou instructions de téléchargement, avec licence, provenance et hashes.
- [x] Générateur local de vidéo de visualisation CEM.
- [x] Animation CEM publiée montrant aussi la trajectoire exécutée.
- [x] Trace de planning compacte, documentée et réutilisable sans relancer l'environnement.
- [x] GIFs réel/prédit, protocole de reconstruction et métriques versionnés.
- [ ] Checkpoints des décodeurs publiés ou reconstruits par une procédure auditée.
- [ ] Résultats représentatifs du checkpoint officiel et du checkpoint local.
- [x] Étude on-policy avec actions exécutées, coût de calcul et cas d'échec.
- [ ] Ablation SIGReg et baseline random shooting.
- [ ] Tableau final des limites : dépendance aux données, écart rollout/réalité, coût de planning et cas d'échec.
- [ ] Licence racine, citation et release versionnée.

## Extensions après la version 1

- probes linéaires complets et analyse du latent ;
- robustesse visuelle et physique hors distribution ;
- iCEM et MPPI ;
- VLA adapté à PushT ou rapport de faisabilité négatif reproductible.

## Hors périmètre initial

Ces éléments ne seront ajoutés qu'après les sorties précédentes :

- déploiement sur un robot réel ;
- utilisation des images décodées comme coût de planning avant validation du décodeur ;
- préentraînement d'un VLA ou d'un modèle fondation depuis zéro ;
- entraînement multi-GPU ;
- benchmark exhaustif de toutes les baselines du papier.
