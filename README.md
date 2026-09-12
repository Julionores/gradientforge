# GradientForge — Régression & classification *from scratch*

[![CI](https://github.com/Julionores/gradientforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Julionores/gradientforge/actions/workflows/ci.yml)

Un moteur de **régression** (simple, polynomiale, multivariée) et de **classification binaire**,
codé entièrement en **NumPy pur** — sans jamais appeler `scikit-learn` pour l'entraînement.
L'objectif : comprendre et démontrer ce qui se passe *sous le capot* d'un modèle linéaire
(matrice de design, fonction de coût, gradient analytique, descente de gradient), plutôt que
d'utiliser ces algorithmes comme des boîtes noires.

> Projet réalisé par **Junior Tsafack Megnekeu** ([blog.jtmcloud.com](https://blog.jtmcloud.com) ·
> [GitHub](https://github.com/Julionores) ·
> [LinkedIn](https://www.linkedin.com/in/junior-tsafack-megnekeu-b673151b9)) — pièce d'un
> portfolio technique orienté Machine Learning / Cloud. Voir aussi
> [`radar-risque-impaye`](https://github.com/Julionores/radar-risque-impaye), un pipeline de
> classification scikit-learn pour prédire un risque de retard de paiement, et
> [`devsecops-pipeline-reference`](https://github.com/Julionores/devsecops-pipeline-reference),
> [`securebank-api`](https://github.com/Julionores/securebank-api),
> [`postgresql-ha-repmgr`](https://github.com/Julionores/postgresql-ha-repmgr),
> [`iso27001-isms-toolkit`](https://github.com/Julionores/iso27001-isms-toolkit),
> [`dynamodb-streams-cdc-pipeline`](https://github.com/Julionores/dynamodb-streams-cdc-pipeline),
> [`aws-troubleshooting-challenge`](https://github.com/Julionores/aws-troubleshooting-challenge),
> [`s3-cross-region-replication`](https://github.com/Julionores/s3-cross-region-replication),
> [`aws-alb-deployment-patterns`](https://github.com/Julionores/aws-alb-deployment-patterns) et
> [`aws-vpc-connectivity-patterns`](https://github.com/Julionores/aws-vpc-connectivity-patterns).
> Ce projet accompagne le module 3 de mon
> [cours Machine Learning & Deep Learning](https://blog.jtmcloud.com/machine-learning/03-algorithmes-from-scratch/).

## Pourquoi ce projet

Utiliser `sklearn.linear_model.LinearRegression` prend une ligne de code. Comprendre *pourquoi*
ce modèle fonctionne — et surtout, savoir diagnostiquer pourquoi un entraînement se passe mal —
demande d'avoir soi-même implémenté la mécanique : la matrice de design, la fonction de coût,
son gradient, et la boucle d'optimisation qui les relie. C'est exactement la compétence qu'un
service managé comme AWS SageMaker Linear Learner suppose acquise chez qui le configure.

## Deux modules, un seul moteur

```
src/gradientforge/
├── regression.py       # GradientForge : matrice de design, MSE, gradient, descente de gradient
└── classification.py   # BoundaryScope : sigmoïde, log-loss, réutilise gradient_descent
```

`classification.py` ne réimplémente pas la descente de gradient : elle réutilise
`gradient_descent()` de `regression.py` en lui passant une fonction de coût et un gradient
différents (`log_loss`/`log_loss_grad` au lieu de `mse_cost`/`mse_grad`) — la preuve que
régression et classification ne sont, mathématiquement, que deux instances du même algorithme
d'optimisation.

## GradientForge — régression

Prévoir la consommation électrique (kWh) d'un data center à partir de sa télémétrie
(charge CPU, RAM, trafic réseau).

```bash
python examples/gradientforge_demo.py
```

```
=== Regression simple (charge CPU -> consommation) ===
theta appris (pente, biais): [185.44164342 530.75850304]
cout initial: 142824.0849159218
cout final: 107.87219847548347
R2: 0.993765396186994

=== Regression polynomiale vs lineaire (donnees non lineaires) ===
R2 polynomial: 0.9976838727313744
R2 lineaire (sous-ajuste): 0.991894935417701

=== Regression multivariee (CPU + RAM + reseau) ===
theta appris [cpu, ram, reseau, biais]: [141.92694678  60.04631352  30.4126122  603.47328392]
R2: 0.9851055565698277
```

Un test (`tests/test_regression.py::test_matches_sklearn_linear_regression`) confirme que ce
moteur converge vers **les mêmes paramètres** que `sklearn.linear_model.LinearRegression` sur
le même jeu de données — la solution est convexe et unique, peu importe qu'on l'atteigne par
descente de gradient itérative ou par résolution matricielle directe.

## BoundaryScope — classification

Classer un flux réseau comme normal ou à risque, à partir de deux métriques (débit, latence).

```bash
python examples/boundaryscope_demo.py
```

```
=== BoundaryScope : classification de risque reseau ===
perte initiale (log-loss): 0.31999106369297786
perte finale (log-loss): 0.004897298166977283
accuracy test (modele): 1.0
accuracy test (baseline classe majoritaire): 0.45
theta appris [debit, latence, biais]: [-2.76965214  4.22615028  0.35222356]
```

Le modèle bat largement la baseline naïve (toujours prédire la classe majoritaire, 45 %
d'accuracy) — la performance de 100 % n'est donc pas un artefact du déséquilibre des classes.

## Un piège réel, documenté tel quel

En préparant ce projet, une première tentative de régression **sans standardiser** la feature
d'entrée a réellement divergé avec un `learning_rate` pourtant modeste :

```
cout iteration 0: 217621.99598663827
cout iteration 50: 8603580719717.093
cout final (iteration 1999): inf
theta final: [-9.94894191e+152 -1.58851509e+151]
```

Ce cas est conservé tel quel comme test de non-régression
(`test_unscaled_features_with_high_learning_rate_diverge`) : il documente *pourquoi* la
standardisation des features n'est pas une étape cosmétique, mais une condition de stabilité
numérique de l'entraînement.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

12 tests couvrent : la construction de la matrice de design, la standardisation, la
non-régression sur le cas de divergence, la convergence et le R² du modèle de régression
(simple, polynomial, multivarié), la décroissance du log-loss et l'accuracy du classifieur
face à sa baseline, et la correspondance avec les modèles équivalents de scikit-learn.

```
============================= 12 passed in 3.11s ==============================
```

## Installation

```bash
conda create -n gradientforge python=3.11
conda activate gradientforge
pip install -r requirements-dev.txt
```

## Structure du projet

```
src/gradientforge/    # Le moteur : regression.py, classification.py
examples/             # Scripts de demonstration (sorties ci-dessus)
tests/                # Suite de tests pytest
```

## Licence

MIT — voir [`LICENSE`](LICENSE). Projet à but pédagogique et de démonstration.
