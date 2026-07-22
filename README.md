# LeWM PushT Planning Lab

Apprendre un world model visuel compact sur PushT, puis rendre le contrôle par *model predictive control* (MPC) observable : rollouts latents, recherche Cross-Entropy Method (CEM), et mesure du contrôle final.

Le projet s'inspire de [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm) et s'appuie sur [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). Il privilégie un modèle entraînable localement sur une RTX 3090 (24 Go) plutôt qu'un modèle fondation.

> État : cadrage du projet. Voir la [roadmap](ROADMAP.md) pour les livrables et critères de réussite.

## Objectif

À partir d'une observation image `o_t`, l'encodeur produit un état latent `z_t`. Un encodeur d'action et un prédicteur apprennent la dynamique :

```text
z_t = E(o_t)
u_t = A(a_t)
ẑ_(t+1) = P(z_t, u_t)
L = L_prediction + λ L_SIGReg
```

À l'exécution, un planificateur CEM échantillonne des séquences d'actions, les déroule dans le modèle latent et minimise leur distance au latent du but. Seule la première action est appliquée, puis le plan est recalculé à l'observation suivante (MPC en boucle fermée).

## Démonstration visée

L'interface de démonstration affichera simultanément :

1. **L'environnement réel** — pousseur, objet en T, cible et trajectoire exécutée.
2. **La population CEM** — candidats, élites, moyenne de la distribution et dispersion à chaque itération.
3. **Les rollouts latents** — embedding courant, embedding objectif, futurs prédits et coût terminal par candidat.
4. **Les métriques** — meilleur coût, moyenne/variance des coûts, temps de planification, erreur de prédiction multi-step, variance des embeddings et taux de réussite.

L'objectif pédagogique est de voir les séquences d'actions d'abord dispersées se concentrer, itération après itération, vers une poussée qui rapproche réellement le T de sa cible.

## Périmètre scientifique

| Question | Mesure attendue |
| --- | --- |
| Le modèle apprend-il une dynamique latente stable ? | pertes, erreur multi-step, variance des embeddings, détection du collapse |
| CEM aide-t-il à contrôler PushT ? | taux de réussite et coût final en MPC fermé |
| L'optimisation itérative aide-t-elle ? | CEM vs random shooting à budget de rollouts égal |
| Les choix de planning importent-ils ? | ablations sur `N ∈ {32,64,128,256,512}` et `H ∈ {4,8,12,16}` |
| Le latent encode-t-il des variables physiques utiles ? | probes pour position/orientation du T, position du pousseur, distance au but et contact |
| Le monde modèle résiste-t-il aux décalages de distribution ? | succès et erreur sous variations visuelles et physiques PushT |

Les comparaisons iCEM et, si les ressources le permettent, MPPI complètent les baselines. La comparaison centrale reste **CEM contre random shooting à budget de modèles égal**.

## Feuille de route

La [roadmap détaillée](ROADMAP.md) définit l'ordre d'implémentation, les dépendances et les critères de validation. L'ordre est volontairement strict : rendre l'évaluation de référence reproductible avant d'entraîner ou de modifier l'architecture.

```text
Checkpoint LeWM → MPC/CEM instrumenté → visualisation → entraînement local
        → baselines et ablations → probes latents → robustesse
```

## Environnement cible

- GPU : NVIDIA RTX 3090, 24 Go VRAM.
- Système : Linux ou WSL2 avec pilote CUDA compatible PyTorch.
- Python : 3.10, environnement géré avec `uv`.
- Données : PushT et checkpoints officiels LeWM.

Le checkpoint pré-entraîné sert à livrer rapidement une démonstration fiable. L'entraînement local permet ensuite de reproduire le résultat, tester SIGReg et observer les échecs de représentation. Le format de données devra être choisi selon le disque disponible : stable-worldmodel indique environ 43 Go pour HDF5 et 13 Go pour LanceDB sur son benchmark PushT.

## Sources et crédits

- [LeWorldModel — code officiel](https://github.com/lucas-maes/le-wm)
- [stable-worldmodel — plateforme, environnements et solveurs](https://github.com/galilai-group/stable-worldmodel)
- [Checkpoint LeWM PushT officiel](https://huggingface.co/quentinll/lewm-pusht)

Le code propre à ce dépôt sera documenté avec ses versions de dépendances, seeds, configurations et résultats afin que chaque graphique et chaque vidéo soit reproductible.

## Métadonnées GitHub proposées

**Description**

`An instrumented LeWorldModel + PushT lab for latent world-model learning, CEM planning, and MPC visualization.`

**Topics**

`world-model`, `model-predictive-control`, `cross-entropy-method`, `cem`, `jepa`, `representation-learning`, `reinforcement-learning`, `robotics`, `pusht`, `pytorch`, `visualization`, `latent-space`
