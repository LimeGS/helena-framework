"""Physical-scale helpers for cross-scroll V4 experiments."""
import math


def voxels_for_um(distance_um, vox_um):
    if distance_um <= 0 or vox_um <= 0:
        raise ValueError("distance_um and vox_um must be positive")
    return max(1, int(round(distance_um / vox_um)))


def columns_for_revolution_fraction(columns_per_revolution, fraction):
    if columns_per_revolution <= 0 or not 0 < fraction <= 1:
        raise ValueError("columns_per_revolution must be positive and fraction in (0,1]")
    return max(1, int(round(columns_per_revolution * fraction)))


def estimate_columns_per_revolution(theta_span_radians, width_columns):
    if theta_span_radians <= 0 or width_columns <= 0:
        raise ValueError("theta span and width must be positive")
    revolutions = theta_span_radians / (2 * math.pi)
    return float(width_columns / revolutions)


def derive_column_sampling(columns_per_revolution, window_fraction=0.024,
                           stride_fraction=0.012, curvature_lag_fraction=0.006,
                           minimum_separation_fraction=0.036):
    """Derive cross-scroll column parameters from fractions of one revolution.

    Defaults approximate PHerc0332's historical 200/100/50/300 settings at
    ~8,484 columns/revolution, while scaling appropriately for other meshes.
    """
    return {
        "window": columns_for_revolution_fraction(columns_per_revolution, window_fraction),
        "stride": columns_for_revolution_fraction(columns_per_revolution, stride_fraction),
        "curvature_lag": columns_for_revolution_fraction(
            columns_per_revolution, curvature_lag_fraction
        ),
        "minimum_separation": columns_for_revolution_fraction(
            columns_per_revolution, minimum_separation_fraction
        ),
    }
