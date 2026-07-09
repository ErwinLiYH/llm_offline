import unittest
from types import SimpleNamespace

import numpy as np

from utils.rollout.artifacts import (
    _ANTMAZE_GLOBAL_FLOOR_RGBA,
    capture_antmaze_global_render_frame,
    capture_render_frame,
)


class FakeCamera:
    def __init__(self):
        self.type = 7
        self.fixedcamid = 3
        self.lookat = np.array([9.0, 8.0, 7.0], dtype=np.float64)
        self.distance = 1.5
        self.elevation = 11.0
        self.azimuth = 22.0


class FakeViewer:
    def __init__(self):
        self.cam = FakeCamera()


class FakeRenderer:
    def __init__(self):
        self.camera_id = 4
        self.viewer = FakeViewer()
        self.viewer_modes = []

    def _get_viewer(self, mode):
        self.viewer_modes.append(mode)
        return self.viewer


class FakeGeom:
    def __init__(self, name):
        self.name = name


class FakeModel:
    def __init__(self, geom_names=None, geom_rgba=None, extent=8.0):
        geom_names = geom_names or []
        if geom_rgba is None:
            geom_rgba = []
        self.ngeom = len(geom_names)
        self._geoms = [FakeGeom(name) for name in geom_names]
        self.geom_rgba = np.asarray(geom_rgba, dtype=np.float64)
        self.stat = SimpleNamespace(extent=float(extent))

    def geom(self, geom_id):
        return self._geoms[int(geom_id)]


class FakeMaze:
    def __init__(self, *, map_width, map_length, maze_size_scaling):
        self.map_width = map_width
        self.map_length = map_length
        self.maze_size_scaling = maze_size_scaling


class FakeEnv:
    def __init__(
        self,
        *,
        env_family,
        renderer=None,
        model=None,
        maze=None,
        render_error=None,
    ):
        self.env_family = env_family
        self.render_error = render_error
        self.render_snapshots = []
        owner = SimpleNamespace(mujoco_renderer=renderer, model=model)
        if env_family == "pointmaze":
            self.unwrapped = SimpleNamespace(point_env=owner, maze=maze)
        elif env_family == "antmaze":
            self.unwrapped = SimpleNamespace(ant_env=owner, maze=maze)
        else:
            self.unwrapped = SimpleNamespace(maze=maze)

    def render(self):
        if self.render_error is not None:
            raise self.render_error

        if self.env_family == "pointmaze":
            owner = self.unwrapped.point_env
        elif self.env_family == "antmaze":
            owner = self.unwrapped.ant_env
        else:
            owner = None

        renderer = getattr(owner, "mujoco_renderer", None) if owner is not None else None
        model = getattr(owner, "model", None) if owner is not None else None
        snapshot = {"direct": renderer is None}
        if renderer is not None:
            cam = renderer.viewer.cam
            snapshot.update(
                {
                    "camera_id": renderer.camera_id,
                    "lookat": np.array(cam.lookat, dtype=np.float64),
                    "distance": cam.distance,
                    "elevation": cam.elevation,
                    "azimuth": cam.azimuth,
                }
            )
        if model is not None and getattr(model, "geom_rgba", None) is not None:
            snapshot["geom_rgba"] = np.array(model.geom_rgba, dtype=np.float64)
        self.render_snapshots.append(snapshot)
        return np.zeros((2, 2, 3), dtype=np.uint8)


class RolloutArtifactsTest(unittest.TestCase):
    def test_pointmaze_capture_uses_map_topdown_camera_and_restores(self):
        renderer = FakeRenderer()
        maze = FakeMaze(map_width=12, map_length=9, maze_size_scaling=1.0)
        env = FakeEnv(env_family="pointmaze", renderer=renderer, maze=maze)
        original_camera_id = renderer.camera_id
        original_cam_state = {
            "type": renderer.viewer.cam.type,
            "fixedcamid": renderer.viewer.cam.fixedcamid,
            "lookat": np.array(renderer.viewer.cam.lookat, dtype=np.float64),
            "distance": renderer.viewer.cam.distance,
            "elevation": renderer.viewer.cam.elevation,
            "azimuth": renderer.viewer.cam.azimuth,
        }

        frames = []
        capture_render_frame(env, frames, env_family="pointmaze")

        self.assertEqual(len(frames), 1)
        self.assertEqual(renderer.viewer_modes, ["rgb_array"])
        snapshot = env.render_snapshots[0]
        self.assertEqual(snapshot["camera_id"], -1)
        np.testing.assert_allclose(snapshot["lookat"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(snapshot["distance"], 19.2)
        self.assertEqual(snapshot["elevation"], -90.0)
        self.assertEqual(snapshot["azimuth"], 90.0)

        self.assertEqual(renderer.camera_id, original_camera_id)
        self.assertEqual(renderer.viewer.cam.type, original_cam_state["type"])
        self.assertEqual(renderer.viewer.cam.fixedcamid, original_cam_state["fixedcamid"])
        np.testing.assert_allclose(renderer.viewer.cam.lookat, original_cam_state["lookat"])
        self.assertEqual(renderer.viewer.cam.distance, original_cam_state["distance"])
        self.assertEqual(renderer.viewer.cam.elevation, original_cam_state["elevation"])
        self.assertEqual(renderer.viewer.cam.azimuth, original_cam_state["azimuth"])

    def test_non_pointmaze_capture_keeps_default_render_path(self):
        env = FakeEnv(env_family="other")
        frames = []

        capture_render_frame(env, frames, env_family="antmaze")

        self.assertEqual(len(frames), 1)
        self.assertTrue(env.render_snapshots[0]["direct"])

    def test_antmaze_global_capture_recolors_floor_only_during_render(self):
        renderer = FakeRenderer()
        model = FakeModel(
            geom_names=["floor", "wall"],
            geom_rgba=[
                [0.8, 0.9, 0.8, 1.0],
                [0.7, 0.5, 0.3, 1.0],
            ],
        )
        maze = FakeMaze(map_width=5, map_length=5, maze_size_scaling=4.0)
        env = FakeEnv(env_family="antmaze", renderer=renderer, model=model, maze=maze)
        original_floor = np.array(model.geom_rgba[0], dtype=np.float64)

        frames = []
        capture_antmaze_global_render_frame(env, frames)

        self.assertEqual(len(frames), 1)
        np.testing.assert_allclose(
            env.render_snapshots[0]["geom_rgba"][0],
            _ANTMAZE_GLOBAL_FLOOR_RGBA,
        )
        np.testing.assert_allclose(model.geom_rgba[0], original_floor)
        self.assertEqual(renderer.camera_id, 4)
        np.testing.assert_allclose(renderer.viewer.cam.lookat, [9.0, 8.0, 7.0])

    def test_antmaze_global_capture_restores_floor_after_render_error(self):
        renderer = FakeRenderer()
        model = FakeModel(
            geom_names=["floor"],
            geom_rgba=[[0.8, 0.9, 0.8, 1.0]],
        )
        env = FakeEnv(
            env_family="antmaze",
            renderer=renderer,
            model=model,
            maze=FakeMaze(map_width=5, map_length=5, maze_size_scaling=4.0),
            render_error=RuntimeError("render failed"),
        )
        original_floor = np.array(model.geom_rgba[0], dtype=np.float64)

        with self.assertRaises(RuntimeError):
            capture_antmaze_global_render_frame(env, [])

        np.testing.assert_allclose(model.geom_rgba[0], original_floor)
        self.assertEqual(renderer.camera_id, 4)
        np.testing.assert_allclose(renderer.viewer.cam.lookat, [9.0, 8.0, 7.0])


if __name__ == "__main__":
    unittest.main()
