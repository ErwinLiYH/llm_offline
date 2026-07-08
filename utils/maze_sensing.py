"""Re-export shim: the canonical sensing implementation lives in `crossmaze.sensing`."""

from crossmaze.sensing import (  # noqa: F401
    CROSSMAZE_OBS_KEY,
    DEFAULT_BOUNDARY_RISK_FRACTION,
    _cell_bounds,
    _cell_center_xy,
    _cell_status,
    _has_new_corner,
    _is_free_cell,
    _near_cell_boundaries,
    _near_opposite_boundary,
    _nearest_free_row_col,
    _neighbor_status,
    _state_matches_meta,
    build_sensing,
    compute_sensing_state,
    obs_xy_to_row_col,
    render_sensing_text,
    sensing_text_from_obs,
)
