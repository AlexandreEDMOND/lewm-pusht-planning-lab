# Erreur on-policy de LeWM sous CEM

## Résultat

Le protocole a été exécuté sur les **24 cas stratifiés préenregistrés** du lien
offline/contrôle, avec le checkpoint officiel, population 300, 30 itérations,
30 élites, horizon de cinq blocs de cinq actions, but à 25 actions et budget de
50 actions. Ce sous-ensemble contient volontairement les trois échecs connus :
ses proportions de succès ne sont **pas** des estimations de performance
populationnelle.

| Condition | Succès (diagnostic stratifié) | Décisions/épisode | Rollouts CEM/épisode |
| --- | ---: | ---: | ---: |
| `receding_horizon=5` | 21 / 24 | 2 | 18 000 (+2 inférences de plan) |
| `receding_horizon=1` | 7 / 24 | 10 | 90 000 (+10 inférences de plan) |

La replanification à chaque bloc n'améliore donc pas ce contrôle : elle coûte
cinq fois plus de rollouts et dégrade fortement ce diagnostic apparié.

Le tableau apparié est 7 succès communs, 14 RH=5 seulement, 0 RH=1 seulement
et 3 échecs communs. RH=5 totalise 191,26 s sur 12 appels MPC batch (7,97 s par
épisode); RH=1, 857,90 s sur 60 appels (35,75 s par épisode). RH=1 utilise
exactement 5× plus de rollouts et 4,49× plus de temps mur observé.

## Erreur à la décision

Les branches sont comparées seulement lorsqu'elles sont exécutées : cinq blocs
pour RH=5, un seul pour RH=1. À RH=5, la MSE latente médiane augmente de
0,0082 à t=5 vers 0,0644 à t=25; l'erreur médiane de position du T augmente de
6,01 à 9,73 px (P95 14,28 à 24,03 px). À RH=1, l'erreur factuelle à t=5 a une
MSE médiane de 0,0105 et une erreur T médiane de 6,38 px (P95 20,89 px).

![Erreurs par horizon](assets/on_policy_cem_errors_by_horizon.png)

La queue d'erreur est substantielle, mais RH=1 ne la réduit pas à cinq actions.
Les GIFs montrent le plan sélectionné, le réel et leur différence :
[succès](assets/on_policy_cem_success.gif) et [échec](assets/on_policy_cem_failure.gif).

## Interprétation

L'erreur on-policy est bien plus informative que l'erreur offline experte :
elle révèle une dérive prononcée à 25 actions. Elle n'explique toutefois pas à
elle seule la chute de RH=1, dont la première erreur médiane reste comparable
à RH=5. Le résultat négatif réfute l'hypothèse simple « replanifier tous les
cinq pas améliore forcément le succès ». Le prochain investissement doit
prioritairement examiner le coût latent et/ou le solveur CEM; `action_block=1`
reste une hypothèse à tester séparément, pas une conséquence établie ici.

## Association descriptive erreur → résultat et limite Gymnasium

Les AUC sont calculées après médiane par épisode, donc RH=1 ne pèse pas cinq
fois plus. À RH=5, l'erreur T factuelle à 25 actions est 9,23 px chez les
succès contre 23,26 px chez les trois échecs (AUC=1,00); la MSE a AUC=0,83. À
RH=1, l'erreur T à cinq actions a AUC=0,55 et le flux exécuté à 25 actions
AUC=0,45. Ce signal RH=5 dépasse l'AUC offline 0,571, mais reste descriptif.

![Erreur et résultat par épisode](assets/on_policy_cem_error_outcome.png)

L'avertissement Gymnasium est réel : `PushT._get_obs()` renvoie des positions
XY du pousseur hors `[0,512]` (environ -800 à 901) et des vitesses négatives,
alors que l'espace déclaré les borne. RH=5 sort dans 6/24 épisodes (2 échecs),
RH=1 dans 9/24 (8 échecs). Les observations brutes sont conservées et cet effet
peut contribuer à certains échecs RH=1, sans les expliquer tous.

## Reproduction et artefacts

```bash
source scripts/_env.sh
uv run --project . python scripts/run_on_policy_cem_error.py --batch-size 4
```

Les données factuelles compactes versionnées sont
[`on_policy_cem_frame_metrics.csv`](results/on_policy_cem_frame_metrics.csv),
[`on_policy_cem_episode_metrics.csv`](results/on_policy_cem_episode_metrics.csv)
et [`on_policy_cem_results.json`](results/on_policy_cem_results.json). Les NPZ
locaux contiennent contexte exact, plan normalisé final, actions physiques,
latents prédits/réels, états et provenance; aucune trace complète de population
n'est dupliquée.
