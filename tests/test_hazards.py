# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for EW-v2 daily hazard labelling (synthetic series exercise code paths only)."""

import numpy as np

from sentinel.eval.hazards import HAZARDS, daily_onsets, issue_labels


def test_onset_requires_clean_prewindow_and_refractory():
    hz = HAZARDS["H1_hypoxia"]
    x = np.array([6, 6, 6, 2.5, 2.8, 6, 6, 2.0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3.5, 6, 6, 6, 2.0])
    ev, on = daily_onsets(x, hz)
    assert ev.sum() == 4                      # days 3, 4, 7, 21
    assert np.flatnonzero(on).tolist() == [3, 21]  # day 7 is within the 7-day refractory window
    x2 = x.copy(); x2[20] = 3.6               # pre-window value not >= 4.0 -> no onset at 21
    assert not daily_onsets(x2, hz)[1][21]


def test_issue_labels_exclude_ongoing_events_and_unknown_outcomes():
    hz = HAZARDS["H1_hypoxia"]
    x = np.array([6, 6, 6, 6, 2.0, 2.0, 6, 6, np.nan, np.nan, np.nan, 6, 6])
    ev, on = daily_onsets(x, hz)
    y, valid = issue_labels(x, ev, on, horizon=2)
    assert y[2] == 1 and y[3] == 1 and y[1] == 0
    assert not valid[4] and not valid[5]      # inside event
    assert not valid[7]                       # (7, 9] mostly missing and y = 0 -> unknown


def test_refractory_continuation_days_are_not_valid_negatives():
    hz = HAZARDS["H1_hypoxia"]  # refractory 7 d
    x = np.array([6.0] * 5 + [2.0] + [6.0] * 20)
    ev, on = daily_onsets(x, hz)
    y, v_old = issue_labels(x, ev, on, horizon=3)
    _, v_new = issue_labels(x, ev, on, horizon=3, refractory_days=hz.refractory_days)
    # event day 5: issue days 6..9 have their whole window (d, d+3] inside 5+7 = day 12 -> invalid
    assert v_old[6:10].all() and not v_new[6:10].any()
    assert v_new[10] and v_new[3]         # window reaches past the refractory period / before the event
    assert (v_new <= v_old).all()
