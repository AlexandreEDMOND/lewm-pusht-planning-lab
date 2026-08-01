# Espace recherche

Ce dossier rassemble les travaux qui vont au-delà de la démonstration principale
du dépôt. Ils ne sont pas nécessaires pour répondre à la question simple du
README — « LeWM peut-il imaginer des futurs PushT et CEM peut-il sélectionner un
plan prometteur ? » — mais ils documentent les limites, les choix techniques et
les pistes de comparaison.

## Ce qui s'y passe

1. **Décodage visuel.** LeWM prédit des embeddings, pas des images. Les
   expériences de décodeurs rendent ces embeddings observables, comparent réel
   et prédit, et séparent l'erreur du décodeur de celle du world model.
2. **Démonstrateur reproductible.** Deux épisodes CEM fixés sont rejoués avec
   traces complètes, hashes, manifeste et contrôles automatisés. Il sert à
   vérifier la chaîne de bout en bout, pas à estimer une performance générale.
3. **Généralisation et contrôle.** Les rollouts sont analysés sur 128 épisodes,
   puis CEM est rejoué sur 24 cas stratifiés. Ces études quantifient la dérive,
   les cas difficiles et le coût de replanification.
4. **Comparaison VLA.** Le protocole décrit comment comparer un futur VLA à
   LeWM+CEM avec les mêmes images, objectifs, actions et épisodes. Aucun
   résultat VLA n'est encore publié.

## Lire les rapports

- [Faisabilité du décodeur visuel](visual_decoder_feasibility.md)
- [Démonstrateur CEM reproductible](cem_reproducible_demo.md)
- [Généralisation des rollouts](rollout_generalization.md)
- [Erreur on-policy sous CEM](on_policy_cem_error.md)
- [Protocole de comparaison VLA](vla_comparison_protocol.md)
- [Rapport de validation](validation_report.md)
- [Audit historique](project_audit_2026-07-30.md)

Les figures associées sont dans [assets/](assets/) et les données versionnées
dans [results/](results/). Elles sont conservées pour la reproductibilité, mais
ne constituent pas de nouvelles promesses du démonstrateur principal.

## Ce qui reste dans le noyau du projet

Le README, les visuels sous `docs/assets/`, le sous-module LeWM et les scripts
de lancement conservent le chemin minimal : encoder une scène, imaginer des
futurs, les rendre visibles et montrer la recherche CEM sur PushT.
