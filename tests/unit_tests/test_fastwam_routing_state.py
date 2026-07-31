import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch


def _load_routing_modules():
    repo = Path(__file__).resolve().parents[2]
    package_name = "fastwam_routing_under_test"
    package = ModuleType(package_name)
    package.__path__ = [
        str(repo / "rlinf/models/embodiment/wam_policy")
    ]
    sys.modules[package_name] = package
    for module_name in ("contracts", "routing_state"):
        full_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            full_name,
            repo / f"rlinf/models/embodiment/wam_policy/{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.routing_state"]


_routing = _load_routing_modules()
PendingRouteTracker = _routing.PendingRouteTracker


def test_first_chunk_forced_idm_then_consumes_emitted_route():
    tracker = PendingRouteTracker()
    env_ids = torch.tensor([10, 20])
    first = tracker.consume(
        env_ids=env_ids,
        reset_mask=torch.tensor([True, True]),
        actor_version=3,
    )
    assert first.route_used.tolist() == [1, 1]
    assert first.route_was_forced.tolist() == [True, True]

    tracker.emit(
        env_ids=env_ids,
        routes=torch.tensor([0, 1]),
        source_chunk_ids=first.chunk_ids,
        episode_ids=first.episode_ids,
        actor_version=3,
    )
    second = tracker.consume(
        env_ids=env_ids,
        reset_mask=torch.tensor([False, False]),
        actor_version=4,
    )
    assert second.route_used.tolist() == [0, 1]
    assert second.route_source_chunk_ids.tolist() == [0, 0]
    assert second.actor_versions.tolist() == [3, 3]


def test_asynchronous_reset_never_consumes_old_episode_decision():
    tracker = PendingRouteTracker()
    env_ids = torch.tensor([1, 2])
    first = tracker.consume(
        env_ids=env_ids,
        reset_mask=torch.tensor([True, True]),
        actor_version=0,
    )
    tracker.emit(
        env_ids=env_ids,
        routes=torch.tensor([0, 0]),
        source_chunk_ids=first.chunk_ids,
        episode_ids=first.episode_ids,
        actor_version=0,
    )
    next_routes = tracker.consume(
        env_ids=env_ids,
        reset_mask=torch.tensor([True, False]),
        actor_version=1,
    )

    assert next_routes.route_used.tolist() == [1, 0]
    assert next_routes.route_was_forced.tolist() == [True, False]
    assert next_routes.episode_ids[0] != first.episode_ids[0]
    assert next_routes.episode_ids[1] == first.episode_ids[1]


def test_route_state_checkpoint_round_trip():
    tracker = PendingRouteTracker()
    first = tracker.consume(
        env_ids=torch.tensor([4]),
        reset_mask=torch.tensor([True]),
        actor_version=2,
    )
    tracker.emit(
        env_ids=torch.tensor([4]),
        routes=torch.tensor([0]),
        source_chunk_ids=first.chunk_ids,
        episode_ids=first.episode_ids,
        actor_version=2,
    )

    restored = PendingRouteTracker()
    restored.load_state_dict(tracker.state_dict())
    route = restored.consume(
        env_ids=torch.tensor([4]),
        reset_mask=torch.tensor([False]),
        actor_version=7,
    )
    assert route.route_used.item() == 0
    assert route.actor_versions.item() == 2


def test_actor_update_discards_pending_route_and_forces_idm_boundary():
    tracker = PendingRouteTracker()
    first = tracker.consume(
        env_ids=torch.tensor([4]),
        reset_mask=torch.tensor([True]),
        actor_version=2,
    )
    tracker.emit(
        env_ids=torch.tensor([4]),
        routes=torch.tensor([0]),
        source_chunk_ids=first.chunk_ids,
        episode_ids=first.episode_ids,
        actor_version=2,
    )

    tracker.force_idm_after_actor_update()
    boundary = tracker.consume(
        env_ids=torch.tensor([4]),
        reset_mask=torch.tensor([False]),
        actor_version=3,
    )

    assert boundary.route_used.item() == 1
    assert boundary.route_was_forced.item()
    assert boundary.route_source_chunk_ids.item() == -1
    assert boundary.actor_versions.item() == 3
    assert boundary.episode_ids.item() == first.episode_ids.item()
    assert boundary.chunk_ids.item() == 1
