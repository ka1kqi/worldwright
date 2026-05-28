"""WorldwrightScene — the only module that imports `genesis`.

Insulates the rest of the codebase from Genesis API churn and exposes a
typed, agent-friendly surface for LLM-generated scene + reward code.
"""

from __future__ import annotations

from typing import Any, Literal

import genesis as gs
import numpy as np

from .handles import (
    ContactInfo,
    EntityHandle,
    FrankaHandle,
    ObjectState,
    RGB,
    SceneState,
    Vec3,
)


Backend = Literal["metal", "cpu", "cuda", "gpu", "amdgpu"]
_BACKEND_MAP = {
    "cpu": "cpu", "metal": "metal", "cuda": "cuda",
    "gpu": "gpu", "amdgpu": "amdgpu",
}


class WorldwrightScene:
    """The world. One per process. Build once, step many.

    Lifecycle:
        scene = WorldwrightScene(backend="metal")
        scene.add_plane(); scene.add_box(...); franka = scene.add_franka()
        scene.build()
        for _ in range(N): scene.step()
        state = scene.state()
        scene.close()
    """

    def __init__(
        self,
        sim_dt: float = 0.01,
        backend: Backend = "metal",
        show_viewer: bool = False,
    ) -> None:
        self._sim_dt = float(sim_dt)
        self._backend_name = backend
        self._show_viewer = bool(show_viewer)
        self._built = False
        self._t: float = 0.0

        gs.init(backend=getattr(gs, _BACKEND_MAP[backend]), logging_level="warning")

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self._sim_dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            show_viewer=self._show_viewer,
        )

        self._entities: dict[str, EntityHandle] = {}
        self._franka: FrankaHandle | None = None
        self._cameras: dict[str, Any] = {}
        self._recording_cameras: list[str] = []

    # ---------- topology ----------
    def _check_unbuilt(self) -> None:
        if self._built:
            raise RuntimeError("Scene already built; topology is frozen")

    def add_plane(self, name: str = "plane") -> EntityHandle:
        self._check_unbuilt()
        ent = self._scene.add_entity(gs.morphs.Plane())
        h = EntityHandle(name, ent)
        self._entities[name] = h
        return h

    def add_box(
        self,
        name: str,
        size: Vec3,
        pos: Vec3,
        color: RGB | None = None,
    ) -> EntityHandle:
        self._check_unbuilt()
        morph_kwargs: dict[str, Any] = {"size": tuple(size), "pos": tuple(pos)}
        morph = gs.morphs.Box(**morph_kwargs)
        surface = None
        if color is not None:
            r, g, b = color
            surface = gs.surfaces.Default(color=(r, g, b, 1.0))
        ent = (
            self._scene.add_entity(morph, surface=surface)
            if surface is not None
            else self._scene.add_entity(morph)
        )
        h = EntityHandle(name, ent)
        self._entities[name] = h
        return h

    def add_mesh(
        self,
        name: str,
        file: str,
        pos: Vec3,
        scale: float = 1.0,
    ) -> EntityHandle:
        self._check_unbuilt()
        ent = self._scene.add_entity(
            gs.morphs.Mesh(file=file, pos=tuple(pos), scale=scale)
        )
        h = EntityHandle(name, ent)
        self._entities[name] = h
        return h

    def add_franka(self, name: str = "franka") -> FrankaHandle:
        self._check_unbuilt()
        if self._franka is not None:
            raise RuntimeError("Scene already has a Franka")
        ent = self._scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        )
        h = FrankaHandle(name, ent)
        self._franka = h
        self._entities[name] = h
        return h

    def add_camera(
        self,
        name: str,
        pos: Vec3,
        lookat: Vec3,
        res: tuple[int, int] = (640, 480),
        fov: float = 35.0,
    ) -> Any:
        self._check_unbuilt()
        cam = self._scene.add_camera(
            res=res, pos=tuple(pos), lookat=tuple(lookat), fov=fov, GUI=False
        )
        self._cameras[name] = cam
        return cam

    # ---------- lifecycle ----------
    def build(self) -> None:
        if self._built:
            return
        self._scene.build()
        if self._franka is not None:
            self._franka._post_build()
        self._built = True

    def step(self) -> None:
        if not self._built:
            raise RuntimeError("Scene not built")
        self._scene.step()
        self._t += self._sim_dt
        for name in self._recording_cameras:
            self._cameras[name].render()

    # ---------- state ----------
    def state(self) -> SceneState:
        if self._franka is None:
            raise RuntimeError(
                "state() requires a Franka in the scene "
                "(no robot-free scenes in M1)"
            )
        f = self._franka
        objects = {
            name: ObjectState(pos=h.get_pos(), quat=h.get_quat())
            for name, h in self._entities.items()
            if name != f.name and not isinstance(h, FrankaHandle)
        }
        return SceneState(
            t=self._t,
            franka_q=f.get_dofs_position(),
            franka_qdot=f.get_dofs_velocity(),
            ee_pos=f.get_ee_pos(),
            ee_quat=f.get_ee_quat(),
            objects=objects,
            contacts=[],  # populated when we need contact-aware predicates
        )

    @property
    def franka(self) -> FrankaHandle:
        if self._franka is None:
            raise RuntimeError("No Franka in scene")
        return self._franka

    @property
    def entities(self) -> dict[str, EntityHandle]:
        return dict(self._entities)

    @property
    def t(self) -> float:
        return self._t

    # ---------- recording ----------
    def start_recording(self, camera: str = "default") -> None:
        if camera not in self._cameras:
            raise KeyError(f"camera {camera!r} not added")
        self._cameras[camera].start_recording()
        if camera not in self._recording_cameras:
            self._recording_cameras.append(camera)

    def stop_recording(
        self, camera: str, save_to_filename: str, fps: int = 60
    ) -> None:
        self._cameras[camera].stop_recording(
            save_to_filename=save_to_filename, fps=fps
        )
        if camera in self._recording_cameras:
            self._recording_cameras.remove(camera)

    # ---------- cleanup ----------
    def close(self) -> None:
        try:
            gs.destroy()
        except Exception:
            pass

    def __enter__(self) -> "WorldwrightScene":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
