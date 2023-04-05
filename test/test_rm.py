import math

import pytest

from cfrainbow import rm
from cfrainbow.utils import (
    normalize_policy_profile,
    normalize_action_policy,
    normalize_state_policy,
)
import random

random.seed(0)


def is_close(dict1, dict2, abs_tol=1e-8):
    return len(dict1) == len(dict2) and all(
        math.isclose(dict1[key], value2, abs_tol=abs_tol)
        for key, value2 in dict2.items()
    )


def uniform_policy(actions):
    return {action: 1.0 / len(actions) for action in actions}


@pytest.fixture(params=[2] + list(range(5, 101, 5)))
def action_setup(request):
    n_actions = request.param
    actions = list(range(n_actions))
    initial_policy = uniform_policy(actions)
    return n_actions, actions, initial_policy


def test_regret_matching(action_setup):
    n_actions, actions, initial_policy = action_setup

    regrets = {
        action: random.random() * random.randint(-1, 1) / n_actions
        for action in actions
    }
    positive_regrets = {action: max(0.0, regret) for action, regret in regrets.items()}
    if (pos_sum := sum(positive_regrets.values())) > rm.unbound._POS_THRESH:
        new_policy = {action: positive_regrets[action] / pos_sum for action in actions}
    else:
        new_policy = uniform_policy(actions)

    rm.regret_matching(initial_policy, regrets)
    assert is_close(initial_policy, new_policy)

    rm.regret_matching_plus(initial_policy, regrets)
    assert is_close(initial_policy, new_policy)
    assert is_close(regrets, positive_regrets)


def test_regret_matching_predictive(action_setup):
    n_actions, actions, initial_policy = action_setup

    utilities = {
        action: random.random() * random.randint(-1, 1) / n_actions
        for action in actions
    }
    expected_utilities = sum(
        u * policy for u, policy in zip(utilities.values(), initial_policy.values())
    )
    regrets = {a: u - expected_utilities for a, u in utilities.items()}
    prediction = utilities
    expected_prediction = sum(
        pred * policy
        for pred, policy in zip(prediction.values(), initial_policy.values())
    )
    predictive_regrets = {
        action: max(0.0, regret + prediction[action] - expected_prediction)
        for action, regret in regrets.items()
    }
    if (pos_sum := sum(predictive_regrets.values())) > rm.unbound._POS_THRESH:
        new_policy = {
            action: predictive_regrets[action] / pos_sum for action in actions
        }
    else:
        new_policy = uniform_policy(actions)
    rm.predictive_regret_matching(prediction, initial_policy, regrets)
    assert is_close(initial_policy, new_policy)


def test_regret_matcher(action_setup):
    n_actions, actions, policy = action_setup

    minimizer = rm.RegretMatcher(actions)
    # assert the default recommendation is a uniform policy
    assert is_close(minimizer.recommend(0), uniform_policy(actions))

    regrets = {
        action: random.random() * random.randint(-1, 1) / n_actions
        for action in actions
    }
    # feed regrets to minimizer in multiple steps
    # (simulating each history contribution for the information state regret)
    for weight in (weights := [random.random() for _ in range(10)]):
        minimizer.observe(0, lambda action: regrets[action] * weight / sum(weights))
        # assert each time that no new recommendation is computed before a change of iteration
        assert is_close(minimizer.recommend(0), uniform_policy(actions))

    # assert that the basic regret matching schedule works and computes the same outputs as manual regret matching
    rm.regret_matching(policy, regrets)
    assert is_close(minimizer.recommend(1), policy)
    assert is_close(minimizer.cumulative_quantity, regrets)


def test_regret_matcher_plus(action_setup):
    n_actions, actions, policy = action_setup

    minimizer = rm.RegretMatcherPlus(actions)
    # assert the default recommendation is a uniform policy
    assert is_close(minimizer.recommend(1), uniform_policy(actions))

    regrets = {
        action: random.random() * random.randint(-1, 1) / n_actions
        for action in actions
    }
    # feed regrets to minimizer in multiple steps
    for weight in (weights := [random.random() for _ in range(10)]):
        minimizer.observe(0, lambda action: regrets[action] * weight / sum(weights))
        # assert each time that no new recommendation is computed before a change of iteration
        assert is_close(minimizer.recommend(0), uniform_policy(actions))

    rm.regret_matching_plus(policy, regrets)

    assert is_close(minimizer.recommend(1), policy)
    assert is_close(minimizer.cumulative_quantity, regrets)


def test_regret_matching_predictive_plus(action_setup):
    n_actions, actions, initial_policy = action_setup

    utilities = {
        action: random.random() * random.randint(-1, 1) / n_actions
        for action in actions
    }
    expected_utilities = sum(
        u * policy for u, policy in zip(utilities.values(), initial_policy.values())
    )
    regrets = {a: u - expected_utilities for a, u in utilities.items()}
    prediction = utilities
    expected_prediction = sum(
        pred * policy
        for pred, policy in zip(prediction.values(), initial_policy.values())
    )
    positive_regrets = {action: max(0.0, regret) for action, regret in regrets.items()}
    predictive_regrets = {
        action: max(0.0, regret + prediction[action] - expected_prediction)
        for action, regret in positive_regrets.items()
    }
    if (pos_sum := sum(predictive_regrets.values())) > rm.unbound._POS_THRESH:
        new_policy = {
            action: predictive_regrets[action] / pos_sum for action in actions
        }
    else:
        new_policy = uniform_policy(actions)
    rm.predictive_regret_matching_plus(prediction, initial_policy, regrets)
    assert is_close(initial_policy, new_policy)
    assert is_close(regrets, positive_regrets)
