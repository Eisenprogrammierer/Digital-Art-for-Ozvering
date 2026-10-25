"""Contracts for the isometric voxel renderer and the world that uses it.

The point of this suite is not that a particular building looks good, but that the
three assets that share the renderer agree about coordinates: five named views,
one projection that can be inverted, continuous world positions, and a sidecar
that carries the height and walkability tables a game needs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen import voxel
from gen.maps import voxel as world
from gen.objects import buildings, props


def _sample_model():
    model = voxel.Model()
    model.add(0, 0, 0, 3, 2, 1.4, "#b58a62")
    model.add(0.4, 0.4, 1.4, 1.2, 1.0, 2.5, "#4c7f9a")
    model.add(2.0, 0.3, 1.4, 2.6, 0.8, 2.0, "#d9c16a")
    return model


def test_the_five_views_have_names_and_a_stable_order():
    model = _sample_model()
    views = voxel.views(model.parts, size=48)
    assert tuple(views) == voxel.VIEWS
    assert voxel.VIEWS == ("north", "west", "south", "east", "top")
    assert all(canvas.w == 48 and canvas.h == 48 for canvas in views.values())
    assert all(float(canvas.a.mean()) > 0.0 for canvas in views.values())


def test_a_view_is_a_pure_function_of_the_model_and_parameters():
    model = _sample_model()
    one = voxel.render(model.parts, size=64, view="east")
    two = voxel.render(model.parts, size=64, view="east")
    assert np.array_equal(one.rgb, two.rgb)
    assert np.array_equal(one.a, two.a)


def test_the_top_view_is_a_plan_and_the_sides_are_not():
    model = _sample_model()
    top = voxel.render(model.parts, size=64, view="top")
    south = voxel.render(model.parts, size=64, view="south")
    assert not np.array_equal(top.rgb, south.rgb)
    assert float(top.a.mean()) > 0.0


def test_each_side_view_selects_the_two_walls_facing_its_camera():
    assert voxel._VIEW_FACES == {
        "south": ("east", "south"),
        "east": ("east", "north"),
        "north": ("west", "north"),
        "west": ("west", "south"),
    }
    for faces in voxel._VIEW_FACES.values():
        assert len(faces) == 2
        assert len(set(faces)) == 2


def test_projected_bounds_contain_every_corner():
    model = _sample_model()
    for view in voxel.VIEWS:
        x0, y0, x1, y1 = voxel.projected_bounds(model.parts, view)
        assert x1 >= x0 and y1 >= y0
        assert math.isfinite(x0 + y0 + x1 + y1)


def test_the_world_height_table_is_zero_on_water_and_positive_on_land():
    f = world.fields(seed=11, cols=18, rows=13)
    z = world.heightmap(f, levels=6)
    assert z.shape == (13, 18)
    assert np.all(z[f["water"]] == 0)
    assert np.all(z[~f["water"]] >= 1)
    assert np.all(z[~f["water"]] <= 6)


def test_world_projection_inverts_and_continuous_positions_keep_fractions():
    canvas, scene = world.render(seed=5, cols=16, rows=11, tile=10.0,
                                 rivers=1, roads=1, hero="spawn")
    assert canvas.w > 0 and canvas.h > 0
    assert scene["movement"] == "continuous"
    assert scene["projection"] == "isometric-axonometric"
    assert len(scene["heightmap"]) == 11
    assert all(len(row) == 16 for row in scene["heightmap"])
    for x, y in ((3.25, 4.75), scene["spawn"], (scene["hero"]["x"], scene["hero"]["y"])):
        sx, sy = world.world_to_screen(scene, x, y, z=2.0)
        back = world.screen_to_world(scene, sx, sy, z=2.0)
        assert np.allclose(back, (x, y), atol=1e-9)
    assert world.walkable_at(scene, *scene["spawn"])
    assert world.height_at(scene, *scene["spawn"]) >= 0.0
    assert isinstance(world.biome_at(scene, *scene["spawn"]), str)


def test_every_building_plans_and_one_turnaround_renders():
    for kind in buildings.KINDS:
        model, meta = buildings.plan(seed=3, kind=kind)
        assert model.parts and meta["kind"] == kind
        assert meta["views"] == list(voxel.VIEWS)
        assert meta["parts"] == len(model.parts)
        assert len(meta["palette"]) >= 6
    grid, meta = buildings.views(seed=3, kind="office", size=48)
    assert tuple(grid) == voxel.VIEWS
    assert meta["floors"] >= 1


def test_every_object_plans_and_the_named_vehicles_render():
    for kind in props.KINDS:
        model, meta = props.plan(seed=3, kind=kind)
        assert model.parts and meta["kind"] == kind
        assert meta["label"]
        assert meta["parts"] == len(model.parts)
    for kind in ("tank", "plane", "lathe", "desk"):
        canvas, meta = props.render(seed=3, kind=kind, size=64)
        assert canvas.w == 64 and canvas.h == 64
        assert meta["kind"] == kind
        assert float(canvas.a.mean()) > 0.0
    grid, meta = props.views(seed=3, kind="tank", size=40)
    assert tuple(grid) == voxel.VIEWS
    assert meta["kind"] == "tank"


def test_the_object_library_covers_machines_vehicles_aircraft_and_furniture():
    assert {"lathe", "milling_machine", "server_rack", "tank", "plane",
            "helicopter", "desk", "chair", "sofa"} <= set(props.KINDS)
    assert len(props.KINDS) >= 30
    assert len(buildings.KINDS) >= 15
