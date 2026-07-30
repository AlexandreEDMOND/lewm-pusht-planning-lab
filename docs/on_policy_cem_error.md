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
