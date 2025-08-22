"""Tests for the PolicyModel icon overlays."""

from ui_new.state.Policy import PolicyModel
from experiments.policy.actions import ACTION_ICONS


def test_plan_icons_align_with_steps() -> None:
    model = PolicyModel()
    model.plan()
    assert len(model.planIcons) == len(model.planSteps)
    for name, icon in zip(model.planSteps, model.planIcons):
        assert ACTION_ICONS[name] == icon
