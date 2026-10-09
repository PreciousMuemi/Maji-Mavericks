Nzoia step-by-step model readiness
==================================

This audit follows `datasets/STEP_BY_STEP_GUIDE.md` and distinguishes working
software from calibration that the supplied files cannot establish.

Hazard engine
-------------

Outcome required: every building has a flood depth for each supplied return period.

Status: working. The engine validates WGS84 coordinates and raster extent, then
samples RP10, RP20, RP50, RP100, RP200 and RP500. Outside-extent points return a
coverage error instead of zero depth. The source rasters are real JRC hazard data.

Vulnerability engine
--------------------

Outcome required: flood depth becomes a documented building damage ratio.

Status: working as a regional reference model. It linearly interpolates the JRC
2017 Africa residential curve. The Africa industrial curve is implemented but can
only be selected for verified industrial building occupancy. Machinery, inventory,
contents and business interruption are not passed through building curves.

Calibration limitation: the four supplied housing classes share the regional
residential curve. The guide describes qualitative differences, but it supplies no
Kenya class factors or claims calibration. Therefore the implementation does not
invent different damage ratios by housing class. Local class calibration remains a
model-development task, not a software failure.

Exposure engine
---------------

Outcome required: model records contain building ID, coordinates, construction
class, floor area and KES TIV.

Status: working. CSV schedules using the supplied schema pass directly. Narrative
offers are mapped conservatively; facility coordinates and aggregate sums are not
copied into individual buildings. Missing inputs are returned in underwriter language.

Financial engine
----------------

Outcome required: building damage becomes ground-up, gross, ceded and net loss.

Status: working. The deterministic order is ground-up damage, per-risk deductible,
per-risk limit, portfolio gross, quota-share cession, catastrophe XOL cession and net
retention. Default runs apply no policy or reinsurance terms. Terms must be explicitly
supplied through an approved scenario; document text is not silently converted into
binding financial terms.

AI intelligence layer
---------------------

Outcome required: AI materially supports ingestion and underwriting decisions.

Status: working. Claude extracts source-cited facts, identifies contradictions and
missing information, selects evidence-grounded concerns and explains deterministic
model results. Claude does not calculate losses or damage ratios.

Results interface
-----------------

Outcome required: an underwriter can see exposure, key return-period results, class
breakdown, limitations and AI findings.

Status: working for validated schedules. The report displays documented financial
figures, RP100 gross loss, engine readiness, missing inputs and analysis. Dashboard
contracts also support the return-period loss curve and building rankings. Narrative
offers without individual coordinates or TIV remain partial by design.

End-to-end evidence
-------------------

The supplied 500-row synthetic Nzoia portfolio passes exposure validation with no
review records. The live API samples all six rasters and returns 500 building results
per scenario. Automated tests cover raster execution, financial order, schema safety,
agent failure handling and dashboard consistency. Synthetic results must never be
presented as an actual insured portfolio.
