# Comparaison LeWM+CEM / VLA

Cette comparaison est obligatoire avant d'affirmer qu'une famille de modèles est
meilleure qu'une autre sur PushT. À ce jour, **aucun résultat VLA n'est publié**
dans ce dépôt : ce document définit le contrat qui rendra la comparaison utile.

## Hypothèse testée

Un VLA compact fine-tuné sur les démonstrations PushT peut-il obtenir un meilleur
succès de contrôle qu'un world model LeWM planifié par CEM, à informations et
budget d'action comparables ?

Le résultat ne doit pas être décidé à l'avance. Si le VLA échoue, coûte trop cher
ou fait moins bien, cela sera rapporté au même titre qu'un gain.

## Entrées et protocole équitable

| Élément | LeWM+CEM | VLA |
| --- | --- | --- |
| Image courante | Oui | Oui |
| Image objectif | Oui | Oui, comme seconde vue visuelle |
| Instruction | N/A | `pousser le T dans la cible` constante |
| État PushT privilégié | Non pour la politique | Non |
| Actions | 2D continues, normalisation documentée | Même espace et même post-traitement |
| Épisodes de test | Ensemble fixé avant le run | Exactement le même ensemble |
| Succès | Définition officielle PushT | Exactement la même |
| Budget | 50 actions maximum | 50 actions maximum |

La seconde vue objectif est essentielle : LeWM reçoit déjà une image objectif.
Un VLA ne doit pas être pénalisé en ne recevant qu'une instruction constante,
mais il ne doit recevoir ni état ni cible supplémentaires auxquels LeWM n'a pas
accès.

## Candidat et exécution attendue

Le candidat retenu est SmolVLA, modèle VLA compact de 450 M paramètres prévu
pour le fine-tuning et les actions continues. La documentation officielle
indique qu'il accepte des vues caméra, un état capteur et une instruction ; pour
la comparaison principale, l'état capteur est volontairement désactivé et les
deux vues sont l'image courante et l'image objectif.

1. Convertir PushT vers LeRobot sans mélanger les épisodes entre entraînement,
   validation et test.
2. Ajouter les deux vues RGB, actions 2D, timestamps, instruction constante et
   seeds ; conserver les actions brutes et normalisées.
3. Fine-tuner le checkpoint VLA retenu, publier config, checkpoint, logs,
   VRAM, temps et licence.
4. Évaluer VLA et LeWM+CEM sur les mêmes départs, objectifs et seeds.
5. Versionner les métriques par épisode, vidéos réussite/échec, tableau de
   comparaison et intervalle de confiance.

SmolVLA nécessite l'installation de LeRobot et un fine-tuning sur le jeu de
données cible ; ce dépôt ne l'ajoute pas implicitement pour éviter de présenter
une dépendance ou un checkpoint non audité comme un résultat. Voir la
[documentation officielle SmolVLA](https://huggingface.co/docs/lerobot/smolvla)
et sa [carte de modèle](https://huggingface.co/lerobot/smolvla_base).

## Critères de publication

Le rapport VLA ne pourra conclure qu'après publication de :

- succès par épisode, moyenne, dispersion et intervalle de confiance ;
- coût final, latence par action, VRAM, taille du modèle et temps de fine-tuning ;
- checkpoint, versions, licences, hashes et split par épisode ;
- mêmes GIFs de réussite et d'échec pour les deux méthodes ;
- limites explicites : instruction constante, dépendance au pré-entraînement et
  équivalence imparfaite entre planning par objectif-image et politique directe.
