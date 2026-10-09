Nzoia deterministic flood model
================================

Model version: `NZOIA-JRC-AFRICA-2017-v1.0.0`

Purpose
-------

This is a deterministic return-period loss model for underwriting screening. It is
not a short-term forecast and it is not a locally calibrated Kenya claims model.

Hazard
------

The engine samples the supplied JRC river-flood depth GeoTIFF at each verified
building coordinate for RP10, RP20, RP50, RP100, RP200 and RP500. Points outside
the raster return `missing_hazard_coverage`; they are never treated as zero depth.

Vulnerability
-------------

The implementation linearly interpolates the published Africa residential curve
in Table 3-1 of Huizinga, de Moel and Szewczyk (2017), *Global flood
depth-damage functions*, JRC105688, DOI 10.2760/16510. The published Africa
industry curve in Table 3-10 is implemented but is used only when industrial
building occupancy has been verified.

The supplied housing classes map to the Africa residential curve without
class-specific modifiers. The dataset guide describes qualitative construction
differences but supplies no calibrated Kenya factors. Applying invented modifiers
would create unsupported loss differences.

Financial order
---------------

For each location and scenario:

1. `ground_up = TIV × damage_ratio`
2. `insured = min(max(ground_up - per_risk_deductible, 0), per_risk_limit)`
3. `gross = sum(insured)`
4. `quota_share_ceded = gross × quota_share`
5. `cat_xol_ceded = min(max(gross - quota_share_ceded - attachment, 0), xol_limit)`
6. `net = gross - quota_share_ceded - cat_xol_ceded`

The standard portfolio run has no deductible, limit or reinsurance terms, so its
reported `total_loss` is gross building loss. Alternative terms must be supplied
explicitly through the scenario interface.

Scope and limitations
---------------------

Only buildings with individual coordinates, KES TIV and a supported housing class
can run. Machinery, inventory, contents and business interruption are excluded.
The calculation has no local floor-height, elevation, flood-defence, duration,
contamination or claims-calibration adjustment. Synthetic exposure results must be
labelled synthetic and must not be presented as a real insured portfolio.
