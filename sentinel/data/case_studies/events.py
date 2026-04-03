"""Historical contamination event definitions for SENTINEL validation.

Defines 10 well-documented contamination events across the United States
with accurate geographic coordinates, dates, and data availability
annotations.  These events serve as ground-truth validation cases for
the SENTINEL anomaly detection system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ContaminationEvent:
    """Metadata for a documented water contamination event."""

    name: str
    event_id: str  # slug, e.g. "gold_king_mine_2015"
    year: int | list[int]  # single year or list for recurring events

    # Geographic extent (WGS-84)
    location_bbox: tuple[float, float, float, float]  # (west, south, east, north)
    center_lat: float
    center_lon: float

    # Contaminant classification
    contaminant_class: str  # nutrient, heavy_metals, thermal, industrial_chemical, ...
    contaminant_detail: str  # specific chemicals

    # Temporal information
    documented_onset: str  # ISO date or "recurring"
    documented_detection: str  # when authorities detected it
    documented_impact: str  # ecological and public-health consequences

    # Data availability flags
    data_availability: dict[str, bool] = field(default_factory=dict)
    epa_response_url: str = ""
    sentinel2_available: bool = False  # Sentinel-2 launched June 2015
    notes: str = ""


# ---------------------------------------------------------------------------
# Historical events catalog
# ---------------------------------------------------------------------------

HISTORICAL_EVENTS: list[ContaminationEvent] = [
    # ------------------------------------------------------------------
    # 1. Gold King Mine spill (2015)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Gold King Mine Spill",
        event_id="gold_king_mine_2015",
        year=2015,
        location_bbox=(-108.05, 37.03, -107.60, 37.45),
        center_lat=37.8930,
        center_lon=-107.6340,
        contaminant_class="acid_mine",
        contaminant_detail=(
            "Iron, lead, arsenic, cadmium, copper, zinc, aluminum, "
            "manganese; acid mine drainage with pH ~3.0"
        ),
        documented_onset="2015-08-05",
        documented_detection="2015-08-05",
        documented_impact=(
            "3 million gallons of acid mine drainage released into Cement Creek "
            "and Animas River. Visible orange plume traveled 130+ miles into the "
            "San Juan River and Lake Powell. Navajo Nation water intakes shut down. "
            "Heavy metals exceeded EPA aquatic life criteria by 100-1000x."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": True,
            "microbial": False,
            "transcriptomic": False,
            "behavioral": False,
        },
        epa_response_url="https://response.epa.gov/site/site_profile.aspx?site_id=14296",
        sentinel2_available=True,
        notes=(
            "EPA-triggered spill during remediation work at the Gold King Mine "
            "near Silverton, CO. USGS station 09358000 (Animas at Farmington) "
            "and 09359020 (San Juan at Four Corners) captured the plume passage."
        ),
    ),

    # ------------------------------------------------------------------
    # 2. Lake Erie harmful algal bloom (annual, 2018-2023)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Lake Erie Harmful Algal Bloom",
        event_id="lake_erie_hab_2018_2023",
        year=[2018, 2019, 2020, 2021, 2022, 2023],
        location_bbox=(-83.60, 41.35, -82.70, 41.80),
        center_lat=41.60,
        center_lon=-83.15,
        contaminant_class="nutrient",
        contaminant_detail=(
            "Microcystis aeruginosa; microcystin-LR; phosphorus and nitrogen "
            "loading from Maumee River agricultural runoff"
        ),
        documented_onset="recurring",
        documented_detection="recurring",
        documented_impact=(
            "Annual cyanobacterial blooms covering 300-700 km² of western "
            "Lake Erie. Fish kills, beach closures, drinking water advisories "
            "for Toledo, OH and surrounding communities. Hypoxic dead zones in "
            "the central basin. Estimated $65M+ annual economic impact."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": True,
            "microbial": True,
            "transcriptomic": False,
            "behavioral": True,
        },
        epa_response_url="https://www.epa.gov/glwqa/western-lake-erie-harmful-algal-blooms",
        sentinel2_available=True,
        notes=(
            "NOAA produces annual Lake Erie HAB severity forecasts. "
            "USGS Maumee River at Waterville (04193500) provides nutrient loading. "
            "Satellite chlorophyll-a from MODIS/Sentinel-2/S3-OLCI available."
        ),
    ),

    # ------------------------------------------------------------------
    # 3. Toledo water crisis (2014)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Toledo Water Crisis",
        event_id="toledo_water_crisis_2014",
        year=2014,
        location_bbox=(-83.70, 41.55, -83.40, 41.75),
        center_lat=41.6528,
        center_lon=-83.5379,
        contaminant_class="nutrient",
        contaminant_detail=(
            "Microcystin-LR cyanotoxin exceeding 1 ppb (WHO guideline); "
            "Microcystis aeruginosa bloom at Collins Park Water Treatment Plant intake"
        ),
        documented_onset="2014-08-01",
        documented_detection="2014-08-02",
        documented_impact=(
            "Do-not-drink order issued for 500,000+ residents in Toledo, OH "
            "and surrounding areas for 3 days. National Guard deployed to "
            "distribute bottled water. Microcystin measured at 2.5 ppb at "
            "finished-water tap (2.5x WHO limit). Precipitated Safe Drinking "
            "Water Act amendments and Ohio nutrient reduction strategy."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": False,
            "microbial": True,
            "transcriptomic": False,
            "behavioral": False,
        },
        epa_response_url="https://www.epa.gov/nutrient-policy-data/toledo-drinking-water-advisory",
        sentinel2_available=False,
        notes=(
            "Pre-Sentinel-2; MODIS and Landsat 8 imagery available. USGS "
            "station 04193500 (Maumee at Waterville) recorded elevated turbidity "
            "and nutrients preceding the event."
        ),
    ),

    # ------------------------------------------------------------------
    # 4. Dan River coal ash spill (2014)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Dan River Coal Ash Spill",
        event_id="dan_river_coal_ash_2014",
        year=2014,
        location_bbox=(-79.75, 36.40, -79.55, 36.60),
        center_lat=36.5002,
        center_lon=-79.6667,
        contaminant_class="heavy_metals",
        contaminant_detail=(
            "Arsenic, selenium, chromium, lead, mercury, thallium, "
            "cadmium; coal combustion residuals (fly ash, bottom ash)"
        ),
        documented_onset="2014-02-02",
        documented_detection="2014-02-02",
        documented_impact=(
            "Stormwater drainage pipe beneath Duke Energy's Dan River Steam "
            "Station collapsed, releasing 39,000 tons of coal ash and "
            "27 million gallons of contaminated water into the Dan River. "
            "Ash deposits detected 70+ miles downstream at Kerr Reservoir. "
            "Arsenic levels exceeded drinking water standards. Duke Energy "
            "fined $102 million (largest Clean Water Act penalty in NC)."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": False,
            "microbial": False,
            "transcriptomic": False,
            "behavioral": True,
        },
        epa_response_url="https://response.epa.gov/site/site_profile.aspx?site_id=12498",
        sentinel2_available=False,
        notes=(
            "Pre-Sentinel-2; Landsat 8 imagery available. USGS station "
            "02075045 (Dan River at Eden, NC) captured turbidity and "
            "conductance anomalies. EPA and NC DEQ monitoring data available."
        ),
    ),

    # ------------------------------------------------------------------
    # 5. Elk River MCHM spill (2014)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Elk River MCHM Chemical Spill",
        event_id="elk_river_mchm_2014",
        year=2014,
        location_bbox=(-81.80, 38.30, -81.50, 38.45),
        center_lat=38.3518,
        center_lon=-81.6964,
        contaminant_class="industrial_chemical",
        contaminant_detail=(
            "4-methylcyclohexanemethanol (MCHM) crude; "
            "dipropylene glycol phenyl ether (PPh); coal-processing chemical"
        ),
        documented_onset="2014-01-09",
        documented_detection="2014-01-09",
        documented_impact=(
            "10,000 gallons of crude MCHM leaked from Freedom Industries tank "
            "into Elk River, 1.5 miles upstream of the West Virginia American "
            "Water intake. Do-not-use order for 300,000 residents in 9 counties "
            "lasting 9 days. Licorice odor reported. 369 ER visits for "
            "exposure symptoms. Highlighted vulnerability of surface water "
            "intakes to upstream chemical storage."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": False,
            "microbial": False,
            "transcriptomic": False,
            "behavioral": False,
        },
        epa_response_url="https://response.epa.gov/site/site_profile.aspx?site_id=12474",
        sentinel2_available=False,
        notes=(
            "Pre-Sentinel-2. USGS station 03197000 (Elk River at Queen Shoals) "
            "and 03198000 (Kanawha River at Charleston) available. Chemical "
            "was poorly characterized — MCHM toxicology data was limited."
        ),
    ),

    # ------------------------------------------------------------------
    # 6. Houston Ship Channel (recurring industrial contamination)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Houston Ship Channel Industrial Contamination",
        event_id="houston_ship_channel_recurring",
        year=[2018, 2019, 2020, 2021, 2022, 2023],
        location_bbox=(-95.30, 29.60, -94.80, 29.85),
        center_lat=29.7355,
        center_lon=-95.0651,
        contaminant_class="petrochemical",
        contaminant_detail=(
            "Benzene, toluene, ethylbenzene, xylene (BTEX); polycyclic "
            "aromatic hydrocarbons (PAHs); dioxins; heavy metals; "
            "wastewater discharges from 200+ industrial facilities"
        ),
        documented_onset="recurring",
        documented_detection="recurring",
        documented_impact=(
            "One of the most industrialized waterways in the US with 200+ "
            "petrochemical facilities. Chronic contamination from permitted "
            "and unpermitted discharges. ITC fire (March 2019) released "
            "benzene plume. Elevated cancer rates in adjacent communities. "
            "Fish consumption advisories for dioxin. San Jacinto Waste Pits "
            "Superfund site adjacent."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": True,
            "microbial": False,
            "transcriptomic": False,
            "behavioral": True,
        },
        epa_response_url="https://response.epa.gov/site/site_profile.aspx?site_id=15368",
        sentinel2_available=True,
        notes=(
            "USGS station 08075000 (Buffalo Bayou at Houston) and TCEQ "
            "monitoring. ITC Deer Park tank fire (March 17, 2019) is a "
            "well-documented acute event within the chronic contamination zone."
        ),
    ),

    # ------------------------------------------------------------------
    # 7. Flint water crisis (2014-2019)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Flint Water Crisis",
        event_id="flint_water_crisis_2014",
        year=[2014, 2015, 2016, 2017, 2018, 2019],
        location_bbox=(-83.75, 42.95, -83.60, 43.10),
        center_lat=43.0125,
        center_lon=-83.6875,
        contaminant_class="heavy_metals",
        contaminant_detail=(
            "Lead (Pb) leaching from service lines and plumbing; "
            "iron; Legionella pneumophila; trihalomethanes (THMs); "
            "E. coli from inadequately treated Flint River water"
        ),
        documented_onset="2014-04-25",
        documented_detection="2015-09-01",
        documented_impact=(
            "Switch from Detroit water (Lake Huron) to Flint River on "
            "April 25, 2014 without corrosion control caused lead leaching "
            "from pipes. 90th-percentile lead levels reached 27 ppb "
            "(1.8x action level). Blood lead levels in children under 5 "
            "doubled. 12 deaths from Legionnaires' disease outbreak. "
            "Federal emergency declared January 2016. $600M+ in "
            "infrastructure replacement."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": True,
            "microbial": True,
            "transcriptomic": False,
            "behavioral": False,
        },
        epa_response_url="https://www.epa.gov/flint",
        sentinel2_available=True,
        notes=(
            "Primarily a drinking water distribution system crisis rather "
            "than surface water contamination, but Flint River water quality "
            "was the root cause. USGS station 04148500 (Flint River at Flint) "
            "has continuous data. Extensive EPA/state monitoring records."
        ),
    ),

    # ------------------------------------------------------------------
    # 8. Gulf of Mexico dead zone (annual hypoxia)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Gulf of Mexico Hypoxic Zone",
        event_id="gulf_dead_zone_annual",
        year=[2018, 2019, 2020, 2021, 2022, 2023],
        location_bbox=(-93.50, 28.00, -89.00, 30.00),
        center_lat=29.00,
        center_lon=-91.25,
        contaminant_class="nutrient",
        contaminant_detail=(
            "Nitrogen (nitrate) and phosphorus from Mississippi-Atchafalaya "
            "River Basin agricultural runoff; seasonal stratification-driven "
            "bottom-water hypoxia (<2 mg/L dissolved oxygen)"
        ),
        documented_onset="recurring",
        documented_detection="recurring",
        documented_impact=(
            "Annual hypoxic zone averaging ~14,000 km² (range 2,100-22,700 km²), "
            "the largest in the Western Hemisphere. Bottom dissolved oxygen "
            "drops below 2 mg/L, causing fish kills and loss of benthic "
            "habitat. Shrimp fishery displacement. Mississippi River/Gulf of "
            "Mexico Hypoxia Task Force target: reduce to <5,000 km² by 2035."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": True,
            "microbial": False,
            "transcriptomic": False,
            "behavioral": True,
        },
        epa_response_url="https://www.epa.gov/ms-htf",
        sentinel2_available=True,
        notes=(
            "LUMCON (Louisiana Universities Marine Consortium) conducts "
            "annual shelf-wide cruise every July. USGS station 07374000 "
            "(Mississippi at Baton Rouge) and 07381490 (Atchafalaya at "
            "Morgan City) track nutrient loads. NOAA annual forecasts available."
        ),
    ),

    # ------------------------------------------------------------------
    # 9. Chesapeake Bay nutrient blooms (seasonal)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="Chesapeake Bay Nutrient Blooms",
        event_id="chesapeake_bay_blooms_seasonal",
        year=[2018, 2019, 2020, 2021, 2022, 2023],
        location_bbox=(-77.30, 36.80, -75.50, 39.60),
        center_lat=38.20,
        center_lon=-76.40,
        contaminant_class="nutrient",
        contaminant_detail=(
            "Nitrogen and phosphorus from agricultural runoff, wastewater "
            "treatment plants, and atmospheric deposition; spring diatom "
            "and summer cyanobacterial blooms; seasonal bottom-water anoxia"
        ),
        documented_onset="recurring",
        documented_detection="recurring",
        documented_impact=(
            "Largest estuary in the US with chronic eutrophication. "
            "Summer dead zone (anoxic volume up to 10 km³). Submerged "
            "aquatic vegetation loss; oyster reef degradation; blue crab "
            "and striped bass habitat compression. Chesapeake Bay TMDL "
            "(2010) mandates 25% nitrogen and 24% phosphorus reduction by 2025."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": True,
            "microbial": True,
            "transcriptomic": False,
            "behavioral": True,
        },
        epa_response_url="https://www.epa.gov/chesapeake-bay-tmdl",
        sentinel2_available=True,
        notes=(
            "Chesapeake Bay Program monitors 150+ stations. USGS stations "
            "on Susquehanna (01578310), Potomac (01646500), and James "
            "(02037500) rivers provide nutrient loading. CBP water quality "
            "database publicly available."
        ),
    ),

    # ------------------------------------------------------------------
    # 10. East Palestine train derailment (2023)
    # ------------------------------------------------------------------
    ContaminationEvent(
        name="East Palestine Train Derailment",
        event_id="east_palestine_derailment_2023",
        year=2023,
        location_bbox=(-80.60, 40.78, -80.45, 40.90),
        center_lat=40.8390,
        center_lon=-80.5184,
        contaminant_class="industrial_chemical",
        contaminant_detail=(
            "Vinyl chloride (intentional burn of 5 tank cars); "
            "butyl acrylate; ethylhexyl acrylate; ethylene glycol "
            "monobutyl ether; isobutylene; phosgene and hydrogen "
            "chloride (combustion products)"
        ),
        documented_onset="2023-02-03",
        documented_detection="2023-02-03",
        documented_impact=(
            "Norfolk Southern train 32N derailed 38 cars in East Palestine, "
            "OH including 11 carrying hazardous materials. Controlled burn "
            "of vinyl chloride on Feb 6 created massive toxic plume. "
            "Contamination of Sulphur Run, Leslie Run, and North Fork "
            "Little Beaver Creek. 3,500+ dead fish within 7.5 miles. "
            "Mandatory evacuation of 1-mile radius. Dioxin-class "
            "byproducts detected in soil. EPA issued Unilateral "
            "Administrative Order to Norfolk Southern."
        ),
        data_availability={
            "usgs_sensors": True,
            "sentinel2": True,
            "microbial": False,
            "transcriptomic": False,
            "behavioral": True,
        },
        epa_response_url="https://response.epa.gov/site/site_profile.aspx?site_id=15933",
        sentinel2_available=True,
        notes=(
            "EPA deployed continuous water monitoring at 20+ locations. "
            "USGS installed temporary gaging stations on Sulphur Run and "
            "Leslie Run. OH EPA conducted extensive fish and macroinvertebrate "
            "surveys. Sentinel-2 imagery captures pre/post burn signature."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_event(event_id: str) -> ContaminationEvent:
    """Retrieve a single event by its ``event_id`` slug.

    Parameters
    ----------
    event_id:
        Event identifier (e.g. ``"gold_king_mine_2015"``).

    Returns
    -------
    The matching :class:`ContaminationEvent`.

    Raises
    ------
    KeyError
        If no event matches the given ID.
    """
    for event in HISTORICAL_EVENTS:
        if event.event_id == event_id:
            return event
    available = [e.event_id for e in HISTORICAL_EVENTS]
    raise KeyError(
        f"Event '{event_id}' not found. Available: {available}"
    )


def get_events_by_class(contaminant_class: str) -> list[ContaminationEvent]:
    """Filter events by contaminant class.

    Parameters
    ----------
    contaminant_class:
        One of ``"nutrient"``, ``"heavy_metals"``, ``"thermal"``,
        ``"industrial_chemical"``, ``"petrochemical"``, ``"sewage"``,
        ``"acid_mine"``.

    Returns
    -------
    List of matching events (may be empty).
    """
    return [
        e for e in HISTORICAL_EVENTS
        if e.contaminant_class == contaminant_class
    ]


def get_events_with_satellite() -> list[ContaminationEvent]:
    """Return only events occurring after Sentinel-2 launch (June 2015).

    Returns
    -------
    Events where ``sentinel2_available`` is *True*.
    """
    return [e for e in HISTORICAL_EVENTS if e.sentinel2_available]
