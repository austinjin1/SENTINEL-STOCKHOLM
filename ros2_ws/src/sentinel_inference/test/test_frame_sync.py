"""Tests for RGB<->NIR timestamp pairing. Pure Python."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinel_inference.frame_sync import ApproxTimeSync


def test_empty_returns_none():
    assert ApproxTimeSync().match(1.0) is None


def test_matches_within_slop():
    s = ApproxTimeSync(slop_sec=0.05)
    s.add(10.00, "a")
    assert s.match(10.02) == "a"


def test_rejects_outside_slop():
    s = ApproxTimeSync(slop_sec=0.05)
    s.add(10.00, "a")
    assert s.match(10.20) is None


def test_picks_nearest():
    s = ApproxTimeSync(slop_sec=0.10)
    s.add(10.00, "early")
    s.add(10.08, "late")
    assert s.match(10.07) == "late"


def test_buffer_evicts_old():
    s = ApproxTimeSync(slop_sec=0.05, max_buffer=2)
    s.add(1.0, "a")
    s.add(2.0, "b")
    s.add(3.0, "c")   # evicts "a"
    assert len(s) == 2
    assert s.match(1.0) is None   # "a" gone, 2.0/3.0 outside slop of 1.0
    assert s.match(3.0) == "c"
