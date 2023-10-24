from ..ALConstruction import _preprocess_scores_linearly, _preprocess_scores_softmax
import pytest


@pytest.mark.parametrize(
    "scores, do_negation, expected",
    [
        (
            {1: 2, 2: 2, 3: 2},
            False,
            {1: 1 / 3, 2: 1 / 3, 3: 1 / 3},
        ),
        (
            {1: 1, 2: 0, 3: 0},
            False,
            {1: 1, 2: 0, 3: 0},
        ),
        ({}, False, {}),
        (
            {1: 0, 2: -2, 3: -2},
            True,
            {1: 0, 2: 0.5, 3: 0.5},
        ),
    ],
)
def test_preprocess_scores_linearly(scores, do_negation, expected):
    result = _preprocess_scores_linearly(scores, do_negation)
    for k, v in expected.items():
        assert result[k] == pytest.approx(v)


@pytest.mark.parametrize(
    "scores, do_negation, divide, divide_factor, expected",
    [
        (
            {1: 2, 2: 2, 3: 2},
            False,
            False,
            None,
            {1: 1 / 3, 2: 1 / 3, 3: 1 / 3},
        ),
        (
            {1: 1, 2: 0, 3: 0},
            False,
            False,
            None,
            {1: 0.5761168847658291, 2: 0.21194155761708544, 3: 0.21194155761708544},
        ),
        (
            {1: 1, 2: 0, 3: 0},
            False,
            True,
            0.5,
            {1: 0.7869860421615985, 2: 0.10650697891920075, 3: 0.10650697891920075},
        ),
        (
            {1: -1, 2: -1, 3: 0},
            True,
            True,
            0.5,
            {1: 0.4683105308334812, 2: 0.4683105308334812, 3: 0.06337893833303762},
        ),
    ],
)
def test_preprocess_scores_softmax(
    scores, do_negation, divide, divide_factor, expected
):
    result = _preprocess_scores_softmax(scores, do_negation, divide, divide_factor)
    for k, v in expected.items():
        assert result[k] == pytest.approx(v)


def test_preprocess_scores_softmax_exception():
    with pytest.raises(AssertionError):
        _preprocess_scores_softmax({1: 2, 2: 3, 3: 4}, False, True, None)
