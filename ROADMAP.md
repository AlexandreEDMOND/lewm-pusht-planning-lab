# Roadmap

Cette roadmap découpe le projet en jalons démontrables. Une phase ne commence que lorsque son critère de sortie est vérifié ; cela évite de confondre problème de modèle, problème de planning et problème de visualisation.

## Décisions de cadrage

- **Tâche initiale** : PushT visuel et actions continues 2D, sans robot réel.
- **Contrôle** : MPC en boucle fermée ; une action appliquée, replanification à chaque pas.
- **Coût principal** : distance entre le latent prédit à l'horizon et le latent de l'image objectif.
- **Référence** : checkpoint LeWM officiel pour la première démo ; entraînement local après validation de la chaîne complète.
- **Matériel cible** : une RTX 3090 de 24 Go. Les expérimentations commencent en précision mixte et avec un batch compatible avec la VRAM mesurée.
- **Comparaison équitable** : les solveurs reçoivent le même budget total de rollouts de modèle.

## Phase 0 — Bootstrap reproductible

**But :** installer et figer un environnement capable de lancer PushT et l'évaluation officielle.

- [ ] Créer le projet Python avec `uv` et verrouiller les dépendances.
- [ ] Installer LeWM/stable-worldmodel et documenter les versions CUDA, PyTorch et driver.
- [ ] Télécharger et valider le dataset PushT ainsi que le checkpoint officiel.
- [ ] Ajouter une configuration locale non versionnée pour les chemins de données et de checkpoints.
- [ ] Enregistrer une commande unique pour une évaluation déterministe (seed fixée).

**Sortie vérifiable :** une commande documentée charge le checkpoint, exécute au moins un épisode PushT et écrit un résultat structuré (seed, configuration, métriques et version du code).

## Phase 1 — Référence MPC avec CEM

**But :** obtenir le contrôle de référence avant toute modification du modèle ou du solveur.

- [ ] Exécuter CEM avec le coût latent et le checkpoint LeWM.
- [ ] Fixer une première configuration : horizon, nombre d'itérations, population, fraction d'élites et bornes d'actions.
- [ ] Évaluer sur un ensemble fixe d'épisodes/seeds.
- [ ] Produire une vidéo ou une suite d'images de quelques épisodes contrôlés.
- [ ] Mesurer taux de réussite, coût final et temps de planning par pas.

**Sortie vérifiable :** les mêmes seeds génèrent les mêmes métriques à une tolérance numérique documentée, et les épisodes sauvegardés montrent une politique qui tente effectivement de placer le T dans la cible.

## Phase 2 — CEM instrumenté

**But :** faire du CEM un objet d'étude plutôt qu'un appel opaque au solveur existant.

- [ ] Définir un format de trace par décision MPC et itération CEM.
- [ ] Enregistrer les séquences candidates, leurs coûts, indices des élites, moyenne, écart-type et temps d'itération.
- [ ] Enregistrer les latents des rollouts et le latent objectif nécessaires à l'analyse.
- [ ] Vérifier que l'instrumentation ne change pas les actions ni le résultat du CEM de référence.
- [ ] Ajouter des tests unitaires sur la sélection des élites, la mise à jour moyenne/écart-type et le respect des bornes d'actions.

**Sortie vérifiable :** pour une décision donnée, une trace permet de reconstruire l'évolution de `μ`, `σ`, les élites et le meilleur coût à chaque itération.

## Phase 3 — Visualisation de planning

**But :** produire une visualisation claire, exportable et fidèle aux traces CEM.

- [ ] Afficher l'environnement réel et la trajectoire exécutée.
- [ ] Afficher la population d'actions et distinguer candidats, élites et moyenne.
- [ ] Afficher les coûts et la contraction de la variance au fil des itérations.
- [ ] Afficher les rollouts latents sans prétendre qu'une projection 2D est une preuve physique.
- [ ] Exporter une vidéo ou un rapport autonome à partir d'une trace sauvegardée.

**Sortie vérifiable :** une vidéo montre, pour un épisode, la population initialement dispersée, la sélection des élites et la concentration de la distribution jusqu'à l'action exécutée. Les chiffres affichés correspondent aux traces brutes.

## Phase 4 — Entraînement local et stabilité de représentation

**But :** reproduire un entraînement LeWM sur la RTX 3090 et mesurer les effets de SIGReg.

- [ ] Lancer l'entraînement LeWM de référence avec logs de `L_prediction`, `L_SIGReg`, VRAM et débit.
- [ ] Sauvegarder checkpoints, config complète et seed.
- [ ] Mesurer l'erreur de prédiction sur plusieurs horizons et la distribution/variance des embeddings.
- [ ] Relancer une expérience sans régularisation pour caractériser le collapse ou son absence.
- [ ] Implémenter une variante VICReg seulement après avoir validé la variante sans régularisation.

**Sortie vérifiable :** chaque run possède un identifiant, une configuration immuable, des courbes de pertes et une évaluation MPC. L'analyse ne conclut à un collapse qu'à partir de plusieurs indicateurs, pas d'une projection 2D seule.

## Phase 5 — Baselines et ablations de planning

**But :** attribuer les gains au planning plutôt qu'au budget de calcul.

- [ ] Implémenter ou instrumenter random shooting avec le même coût latent et le même budget total de rollouts que CEM.
- [ ] Évaluer iCEM avec le même protocole ; ajouter MPPI seulement si la comparaison reste lisible.
- [ ] Balayer `N ∈ {32,64,128,256,512}`.
- [ ] Balayer `H ∈ {4,8,12,16}`.
- [ ] Rapporter moyenne, dispersion, temps de planning et taux de réussite sur les mêmes seeds.

**Sortie vérifiable :** chaque figure/tableau annonce explicitement le budget de rollouts, l'horizon, les seeds et le modèle utilisé. CEM et random shooting sont comparables sans ambiguïté.

## Phase 6 — Analyse du latent

**But :** vérifier quantitativement que le latent encode des variables utiles au contrôle.

- [ ] Générer ou récupérer les cibles de vérité terrain : pose du T, pose du pousseur, distance au but et contact.
- [ ] Geler l'encodeur et séparer les données train/validation/test par épisode.
- [ ] Entraîner des probes légers linéaires, puis documenter leur métrique appropriée (erreur de position, erreur angulaire, précision contact).
- [ ] Comparer checkpoint officiel, LeWM entraîné localement et variante sans SIGReg.

**Sortie vérifiable :** un tableau de probes hors échantillon est produit avec protocole de séparation explicite ; aucune donnée du test n'est utilisée pour sélectionner le probe.

## Phase 7 — Robustesse hors distribution

**But :** tester si contrôle et représentation résistent à des variations pertinentes.

- [ ] Identifier les facteurs de variation PushT disponibles dans stable-worldmodel.
- [ ] Définir des protocoles séparés pour apparence (textures/couleurs) et dynamique.
- [ ] Évaluer le modèle et le contrôleur sur les mêmes seeds entre conditions nominales et OOD.
- [ ] Rapporter réussite, coût final, erreur multi-step et éventuellement signal de surprise.

**Sortie vérifiable :** les résultats distinguent sans ambiguïté variation visuelle et variation physique, avec configurations versionnées.

## Livrables finaux

- [ ] Code d'installation et de reproduction par commandes documentées.
- [ ] Checkpoints ou instructions de téléchargement, avec licence et provenance.
- [ ] Interface ou générateur de vidéo de visualisation CEM.
- [ ] Traces de planning réutilisables sans relancer l'environnement.
- [ ] Rapport de résultats : référence, baselines, ablations, probes et robustesse.
- [ ] Tableau de limites : dépendance au dataset, écart rollout latent/réalité, coût de planning et cas d'échec.

## Hors périmètre initial

Ces éléments ne seront ajoutés qu'après les sorties précédentes :

- déploiement sur un robot réel ;
- reconstruction/décodage pixel des rollouts latents ;
- modèles fondation ou entraînement multi-GPU ;
- benchmark exhaustif de toutes les baselines du papier.
