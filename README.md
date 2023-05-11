<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/cfrainbow_readme_banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/cfrainbow_readme_banner_light.png">
  <img alt="Readme banner." src="docs/cfrainbow_readme_banner_light.png">
</picture>

---

# <p align=center> CFRainbow
<p align=center> Implementations Of <i>Counterfactual Regret Minimization</i> In Its Many Shapes & Colors

[![Python CI](https://github.com/maichmueller/cfrainbow/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/maichmueller/cfrainbow/actions/workflows/python.yml)
[![codecov](https://codecov.io/gh/maichmueller/cfrainbow/branch/main/graph/badge.svg?token=eDon471w2m)](https://codecov.io/gh/maichmueller/cfrainbow)

CFRainbow provides implementations of the basic CFR algorithm and some of the improved variants of CFR
for computing <b>Nash Equilibria</b> in the 2-player 0-sum case as well as <b>Correlated Equilibria</b> in the general-sum case.

The pacakge is loosely built for modularity and general applicability by building on the Openspiel framework by Deepmind. CFRainbow is <b>not</b> built
for performance and will not scale well to large game implementations. Most algorithms are implemented with basic python data structures. 

The package is in the early WIP phase.


# Usage

The easiest way to use CFRainbow is to import the desired algorithm from the package, a regret minimizer, and the `run` method with a pyspiel game object or its OpenSpiel name as input.
For example, to use the Vanilla CFR algorithm:

```python

import cfrainbow
from cfrainbow import cfr, rm

cfrainbow.run(
  cfr.VanillaCFR, 
  n_iter=1000,
  game="kuhn_poker", 
  regret_minimizer=rm.RegretMatcher
)
```
This will run the algorithm for the number of iterations given and compute the exploitability.

You can also use the solver object directly and call `iterate` on it after instantiating it with the correct arguments. 
For example:

```python
from cfrainbow import cfr, rm
from cfrainbow.utils import load_game, KuhnPolicyPrinter, normalize_policy_profile

root_state = load_game("kuhn_poker").new_initial_state()

solver = cfr.VanillaCFR(
    root_state,
    regret_minimizer_type=rm.RegretMatcher,
    alternating=True,  # whether to do alternating or simultaneous updates
)
# iterate for a given number of iterations. 
for i in range(1000):
    solver.iterate()

avg_policy = normalize_policy_profile(solver.average_policy())

print(
    KuhnPolicyPrinter(digits=2).print_profile(avg_policy)
)
```
Output (Infostate --> Action Policy):
```
PLAYER 0:
P1: Jack  | P2:   ?   | cb   --> ['check:  1.00', 'bet:  0.00']
P1: Jack  | P2:   ?          --> ['check:  0.81', 'bet:  0.19']
P1: Queen | P2:   ?   | cb   --> ['check:  0.48', 'bet:  0.52']
P1: Queen | P2:   ?          --> ['check:  0.99', 'bet:  0.01']
P1: King  | P2:   ?   | cb   --> ['check:  0.00', 'bet:  1.00']
P1: King  | P2:   ?          --> ['check:  0.42', 'bet:  0.58']
PLAYER 1:
P1:   ?   | P2: Queen | c    --> ['check:  0.99', 'bet:  0.01']
P1:   ?   | P2: Queen | b    --> ['check:  0.66', 'bet:  0.34']
P1:   ?   | P2: King  | c    --> ['check:  0.00', 'bet:  1.00']
P1:   ?   | P2: King  | b    --> ['check:  0.00', 'bet:  1.00']
P1:   ?   | P2: Jack  | c    --> ['check:  0.67', 'bet:  0.33']
P1:   ?   | P2: Jack  | b    --> ['check:  1.00', 'bet:  0.00']
```
Note that in <i>alternating</i> updates each iteration is a single player's policy update. 
In <i>simultaneous</i> updates both players' policy updates constitute one iteration.


## Available Algorithms

The following list shows the available algorithms that have been implemented (✔️), those that are still work in progress (🔨👷‍♂️),
and those that are planned to be implemented in the future (📅):

| Algorithm | Status | Convergence Tests | Paper Results Reproducing |
|-----------|:--------:|:-------------------:|:--------------------------:|
| [Best-Response CFR](https://www.cs.cmu.edu/~kwaugh/publications/johanson11.pdf) |  ✔️ | ✔️ | 🔨👷‍ |
| [Discounted CFR](https://arxiv.org/abs/1809.04040) | ✔️ | ✔️ | 🔨👷‍ |
| [Exponential CFR](https://arxiv.org/abs/2008.02679) | ✔️ | ✔️ | ❌ |
| [Internal CFR](https://proceedings.neurips.cc/paper/2020/file/5763abe87ed1938799203fb6e8650025-Paper.pdf) | 🔨👷‍♂️ | 🔨👷‍ | 🔨👷‍ |
| [Joint-Reconstruction CFR](https://proceedings.neurips.cc/paper/2019/file/525b8410cc8612283c9ecaf9a319f8ed-Paper.pdf) | 🔨👷‍♂️ | 🔨👷‍ | 🔨👷‍ |
| [Chance Sampling Monte Carlo CFR](http://mlanctot.info/files/papers/nips09mccfr.pdf) | ✔️ | ✔️ | 🔨👷‍ |
| [External Sampling Monte Carlo CFR](http://mlanctot.info/files/papers/nips09mccfr.pdf) | ✔️ | ✔️ | 🔨👷‍ |
| [Outcome Sampling Monte Carlo CFR](http://mlanctot.info/files/papers/nips09mccfr.pdf) | ✔️ | ✔️ | 🔨👷‍ |
| [Predictive Plus CFR](https://arxiv.org/abs/1902.04982) | ✔️ | ✔️ | 🔨👷‍ |
| [Pure CFR](https://richardggibson.appspot.com/static/work/thesis-phd/thesis-phd-paper.pdf) | ✔️ | ✔️ | 🔨👷‍ |
| [Sampling CFR](https://proceedings.neurips.cc/paper/2019/file/525b8410cc8612283c9ecaf9a319f8ed-Paper.pdf) | ✔️ | 🔨👷‍ | 🔨👷‍ |
| [Vanilla CFR](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf) | ✔️ | ✔️ | 🔨👷‍ |
| [Lazy CFR](https://arxiv.org/pdf/1810.04433v3.pdf) | 📅 | 📅 | 📅 |

# Installation

### <b> Web Install </b>

<b> Pip </b>

To install CFRainbow from PyPi using pip, run the following command:
```bash
pip install cfrainbow
```

<b> Poetry </b>

If you're using Poetry to manage your Python packages, you can add CFRainbow to your project by adding the following to your pyproject.toml file:

```toml
[tool.poetry.dependencies]
cfrainbow = "*"
```

Then run:

```bash
poetry install
```

to install your own package. Poetry will then install `cfrainbow` as dependency alongside your own package.
### <b> Source Install </b>

To install CFRainbow from master, please follow these steps:

```bash
git clone https://github.com/maichmueller/cfrainbow.git
cd cfrainbow
```
and install the package with pip
```bash
pip install .
```
or poetry
```bash
poetry install
```
use the option `--no-dev` to ensure a non-editable installation.


