import argparse
import collections
import json
from collections.abc import Callable
from pathlib import Path

import agasc
import numpy as np
import plotly.graph_objects as go
from acdc.common import send_mail
from astropy.table import Table, vstack
from chandra_aca.aca_image import get_aca_images
from chandra_aca.transform import mag_to_count_rate
from cheta import fetch
from cxotime import CxoTime, CxoTimeLike
from jinja2 import Template
from kadi import events
from kadi.commands import get_starcats
from plotly.subplots import make_subplots
from ska_helpers.logging import basic_logger
from ska_helpers.run_info import log_run_info
from ska_numpy import interpolate

from aca_hi_bgd import __version__

# Treat RuntimeWarnings as exceptions
# warnings.filterwarnings("error", category=RuntimeWarning)

DETECT_HITS = 3
DETECT_WINDOW = 21  # seconds
SIX_THRESHOLD = 580  # applied to BGDAVG scaled to e-/s
EIGHT_THRESHOLD = 140  # applied to 8th outer min scaled to e-/s

DOC_ID = "1qhF7tYA4tD3cugsqrtS2jsGe7OSPpIs8jnU4v0f6NCE"
GID = 0
url_start = "https://docs.google.com/spreadsheets/d"
GSHEET_URL = f"{url_start}/{DOC_ID}/export?format=csv&id={DOC_ID}&gid={GID}"
GSHEET_USER_URL = f"{url_start}/{DOC_ID}/edit?usp=sharing"


LOGGER = basic_logger(__name__, level="INFO")


def get_opt():
    parser = argparse.ArgumentParser(description="High Background event finder")
    parser.add_argument("--start", help="Start date")
    parser.add_argument("--stop", help="Stop date")
    parser.add_argument(
        "--data-root",
        default="/proj/sot/ska/data/aca_hi_bgd_mon_dev",
        help="Output data directory",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Replot known events",
    )
    parser.add_argument(
        "--web-out", default="/proj/sot/ska/www/ASPECT/aca_hi_bgd_mon_dev/"
    )
    parser.add_argument(
        "--web-url", default="https://cxc.harvard.edu/mta/ASPECT/aca_hi_bgd_mon_dev"
    )
    parser.add_argument(
        "--email",
        action="append",
        dest="emails",
        default=[],
        help="Email address for notificaion",
    )
    return parser


def get_false_positives() -> Table:
    """
    Get the known false positives from the Google Sheet.

    False positives are labeled on the sheet as type "false positive".

    Parameters
    ----------
    None

    Returns
    -------
    dat : astropy.table.Table
        Table of false positives
    """
    LOGGER.info(f"Reading known false positives {GSHEET_URL}")
    dat = Table.read(GSHEET_URL, format="ascii.csv")
    ok = dat["type"] == "false positive"
    return dat[ok]


def get_extra_notes() -> Table:
    """
    Get any extra notes from the Google Sheet.

    Notes are labeled on the sheet as type "note".

    Parameters
    ----------
    None

    Returns
    -------
    dat : astropy.table.Table
        Table of notes if any
    """
    LOGGER.info(f"Reading notes from {GSHEET_URL}")
    dat = Table.read(GSHEET_URL, format="ascii.csv")
    ok = dat["type"] == "note"
    return dat[ok]


def exceeds_threshold(slot_data: Table) -> np.ndarray:
    """
    Get the indices of the slot data that exceed the threshold.

    This also checks that the image function is 1 (tracking), the image size is not 4 (not useful),
    and the AAPIXTLM is ORIG (not dynamic background data).

    Parameters
    ----------
    slot_data : astropy.table.Table
        Table of slot data with columns "IMGFUNC", "IMGSIZE", "AAPIXTLM", "bgd", "threshold"

    Returns
    -------
    hits : numpy.ndarray
        Boolean array of the slot data that exceeds the threshold
    """
    ok = (
        (slot_data["IMGFUNC"] == 1)
        & (slot_data["IMGSIZE"] != 4)
        & (slot_data["AAPIXTLM"] == "ORIG")
    )
    hits = np.zeros(len(slot_data), dtype=bool)
    hits[ok] = slot_data["bgd"][ok] > slot_data["threshold"][ok]
    return hits


def get_raw_events(
    slots_data: dict, detect_hits: int = None, detect_window: float = None
):
    """
    Find intervals where the background exceeds the threshold.

    This uses a technique of finding intervals where the background exceeds the threshold
    for a certain number of hits in a time window. The default is 3 hits in 21 seconds, which
    are defined with the DETECT_HITS and DETECT_WINDOW global constants.  Those "hits"
    can be in one slot or across multiple slots.

    Parameters
    ----------
    slots_data : dict
        A dictionary of slot data tables with keys 0-7 and tables with columns
        "TIME", "IMGFUNC", "IMGSIZE", "IMGNUM", "AAPIXTLM", "bgd", "threshold"
    detect_hits : int
        The number of hits in the detect_window to consider an event
    detect_window : float
        The time window to consider for an event

    Returns
    -------
    events : list
        List of dictionaries with keys "tstart" and "tstop" for each event
    """
    if detect_hits is None:
        detect_hits = DETECT_HITS
    if detect_window is None:
        detect_window = DETECT_WINDOW

    # Check for background crossings
    events = []

    hit_times = []
    for slot in range(8):
        slot_data = slots_data[slot]
        hits = exceeds_threshold(slot_data)
        if np.count_nonzero(hits) == 0:
            continue
        hit_times.extend(slot_data["TIME"][hits])

    if len(hit_times) < detect_hits:
        return events

    hit_times = np.sort(np.array(hit_times))

    event_start = None
    last_check_time = None
    for time in np.unique(hit_times):
        tok = (hit_times >= time) & (hit_times < time + detect_window)
        count = np.count_nonzero(tok)
        if event_start is None and count >= detect_hits:
            event_start = time
            last_check_time = time
        elif event_start is not None and count >= detect_hits:
            last_check_time = time
        if event_start is not None and count < detect_hits:
            events.append({"tstart": event_start, "tstop": last_check_time})
            event_start = None

    return events


def combine_events(events: list, tol: float = 30) -> list:
    """
    Combine overlapping event intervals.

    This function takes a list of event intervals and returns a reduced list where overlapping or
    close to overlapping intervals are merged.  The tolerance for merging is defined by the tol
    parameter.

    Parameters
    ----------
    events : list
        List of dictionaries with keys "tstart" and "tstop" for each event
    tol : float
        The tolerance for merging overlapping events

    Returns
    -------
    merged_events : list
        List of dictionaries with keys "tstart" and "tstop" for each event
        where overlapping events have been merged
    """
    merged_events = []
    for event in events:
        if len(merged_events) == 0:
            merged_events.append(event)
            continue
        last_event = merged_events[-1]
        if (event["tstart"] - tol) <= last_event["tstop"]:
            last_event["tstop"] = event["tstop"]
        else:
            merged_events.append(event)
    return merged_events


def calculate_slot_seconds(imgsize: np.ndarray) -> float:
    """
    Calculate a metric that adds up the time for each slot where the background is high.

    While the time between images (based on image size) is not really related to the time of the
    high background hit, using the time between images as a "chunk" that can be summed to get a
    metric of how long the background is high.

    Parameters
    ----------
    imgsize : numpy.ndarray
        Array of image sizes (4, 6, and 8)

    Returns
    -------
    slot_seconds : float
        The sum of the time for each slot where the background is high

    """
    dts = np.ones(len(imgsize)) * 1.025
    not4 = imgsize != 4
    dts[not4] = (imgsize[not4] - 4) * 1.025
    return np.sum((dts))


def get_event_stats(event: dict, slots_data: dict) -> dict:
    """
    Get some more background stats info for a high background event.

    Parameters
    ----------
    event : dict
        The event dictionary with at least keys "tstart" and "tstop"
    slots_data : dict
        A dictionary of slot data tables with keys 0-7 and tables with columns
        "TIME", "IMGFUNC", "IMGSIZE", "IMGNUM", "AAPIXTLM", "bgd", "threshold"

    Returns
    -------
    stats : dict
        Dictionary of stats for the event including "s0"-"s7" for each slot, "n_slots", "max_bgd",
        "slot_seconds", "tstop", "datestop", "duration".  The "s0"-"s7" values are 1 if the slot
        is involved in the event and 0 if not.
    """

    stats = {
        "s0": 0,
        "s1": 0,
        "s2": 0,
        "s3": 0,
        "s4": 0,
        "s5": 0,
        "s6": 0,
        "s7": 0,
    }

    cols = ["TIME", "IMGFUNC", "IMGSIZE", "IMGNUM", "AAPIXTLM", "bgd", "threshold"]
    event_data = []
    for slot in range(8):
        slot_data = slots_data[slot]
        tok = (slot_data["TIME"] >= event["tstart"]) & (
            slot_data["TIME"] <= event["tstop"]
        )
        event_data.append(slot_data[cols][tok])
    event_data = vstack(event_data)
    event_data.sort("TIME")

    count_ok = (
        (event_data["IMGFUNC"] == 1)
        & (event_data["IMGSIZE"] != 4)
        & (event_data["AAPIXTLM"] == "ORIG")
        & (event_data["bgd"] > event_data["threshold"])
    )

    stats["max_bgd"] = np.max(event_data["bgd"])

    # Use the image size to default a "time" of the high background hit
    # (even if the frequency at which we get images doesn't really have anything
    # to do with the time of the hit)
    imgsize = event_data["IMGSIZE"].data[count_ok]
    slot_seconds = calculate_slot_seconds(imgsize)
    stats["slot_seconds"] = slot_seconds

    unique_slots = np.unique(event_data["IMGNUM"][count_ok])
    stats["n_slots"] = len(unique_slots)
    for slot in unique_slots:
        stats[f"s{slot}"] = 1

    # update the tstop to include an extra image's worth of time
    last_img = event_data["IMGSIZE"][count_ok][-1]
    stats["tstop"] = event["tstop"] + (last_img - 4) * 1.025
    stats["datestop"] = CxoTime(stats["tstop"]).date
    stats["duration"] = stats["tstop"] - event["tstart"]
    return stats


def get_previous_events(dwell_metrics_file: Path) -> list:
    """
    Get the previous events from the dwell metrics file.
    """
    if (dwell_metrics_file).exists():
        dwell_metrics_tab = Table.read(dwell_metrics_file)
        # Convert the table to a list of dictionaries
        names = dwell_metrics_tab.colnames
        dwell_metrics = [
            dict(zip(names, row, strict=False)) for row in dwell_metrics_tab
        ]
    else:
        dwell_metrics = []
    return dwell_metrics


def get_manvr_extra_data(start: CxoTimeLike, stop: CxoTimeLike) -> dict:
    """
    Get extra data for a dwell

    :param dwell_events: list of dictionaries of events
    :returns: dict of extra data
    """

    pitchs = fetch.Msid("DP_PITCH", start, stop, stat="5min")
    if len(pitchs.vals) > 0:
        pitch = np.median(pitchs.vals)
    else:
        pitch = -999

    guide_cat = get_guide_cat(start)
    expected_acqs = 0
    if "MON" in guide_cat["type"]:
        expected_acqs = 1

    # Check for multiple acquisitions and add notes
    # An extra acquisition is expected for monitor window catalogs
    notes = []
    aoacaseq = fetch.Msid("AOACASEQ", start, stop)
    from cheta.utils import state_intervals

    aca_states = state_intervals(aoacaseq.times, aoacaseq.vals)
    if np.count_nonzero(aca_states["val"] == "AQXN") > expected_acqs:
        notes.append("Full REACQ")

    # Check for NSUN
    # from kadi import events
    # full_dwell = events.dwells.filter(start__exact=dwell.start)[0]
    aopcadmd = fetch.Msid("AOPCADMD", start, CxoTime(stop).secs + 150)
    if aopcadmd.vals[-1] == "NSUN":
        notes.append("NSUN")
    aoacaseq = fetch.Msid("AOACASEQ", stop, CxoTime(stop).secs + 150)
    if aoacaseq.vals[-1] == "BRIT":
        notes.append("BRIT")

    if len(notes) > 0:
        str_notes = ", ".join(notes)
    else:
        str_notes = " " * 21

    return {"pitch": pitch, "notes": str_notes}


def get_events(  # noqa: PLR0912, PLR0915 too many statements, too many branches
    start: CxoTimeLike, stop: CxoTimeLike = None, outdir: Path = "."
) -> tuple:
    """
    Get high background events in a time range.

    Loop over the kadi maneuvers in a time range and check for high background
    events using BGDAVG data and image data.

    Parameters
    ----------
    start : str
        Start of Kalman for the maneuvers to check
    stop : str
        Stop of Kalman for the maneuvers to check

    Returns
    -------
    tuple (bgd_events, time_last_processed)
        bgd_events : astropy.table.Table
            Table of high background events
        time_last_processed : CxoTime
    """
    start = CxoTime(start)
    stop = CxoTime(stop)
    manvrs = events.manvrs.filter(
        kalman_start__gte=start.date, kalman_start__lte=stop.date, n_dwell__gt=0
    )

    bgd_events = []
    if len(manvrs) == 0:
        return bgd_events, None

    dwell_metrics = get_previous_events(Path(outdir) / "dwell_metrics.csv")
    dwell_starts = [d["dwell"] for d in dwell_metrics]
    time_last_processed = None

    bads = get_false_positives()

    for manvr in manvrs:
        try:
            obsid = manvr.obsid
        except ValueError:
            LOGGER.info(f"Skipping manvr {manvr}.")
            continue

        LOGGER.info(
            f"Processing manvr {manvr} obsid {obsid} kalman_start {manvr.kalman_start}"
        )

        if manvr.obsid == 0:
            LOGGER.info(f"Skipping manvr {manvr} obsid {obsid} in anomaly or recovery.")
            continue

        if manvr.kalman_start in bads["dwell_datestart"]:
            LOGGER.info(
                f"Skipping manvr {manvr} obsid {obsid} in known false positives."
            )
            continue

        LOGGER.info(
            f"Processing manvr {manvr} obsid {obsid} kalman_start {manvr.kalman_start}"
        )
        # Do a tiny bit of optimizing - if we already have values for the dwell
        # and none of the slots are above threshold, skip looking for events
        if manvr.kalman_start in dwell_starts:
            row_dict = dwell_metrics[dwell_starts.index(manvr.kalman_start)]
            hit = False
            for slot in range(8):
                if row_dict[f"max_s{slot}"] > row_dict[f"threshold_s{slot}"]:
                    hit = True
                    break
            if hit is False:
                LOGGER.info(f"Skipping manvr {manvr} obsid {obsid} no hits")
                time_last_processed = manvr.next_nman_start
                continue

        start = manvr.kalman_start
        stop = manvr.next_nman_start
        obsid = manvr.obsid
        # If this dwell has a transition to NSUN or NMAN, stop the review at that point
        # and don't process the rest of the dwell
        aopcadmd = fetch.Msid("AOPCADMD", start, stop)
        if np.any(aopcadmd.vals != "NPNT"):
            stop = aopcadmd.times[np.where(aopcadmd.vals != "NPNT")[0][0]]
        # try:
        dwell_events, dwell_end_time, slot_metrics = get_manvr_events(
            start, stop, obsid
        )

        # except IndexError:
        #    LOGGER.info(f"Skipping dwell {d} obsid {d.get_obsid()} no data")
        #    continue
        if dwell_end_time is not None:
            time_last_processed = dwell_end_time

        # save per slot data
        dwell_metric = {"dwell": start, "obsid": obsid}
        for slot in slot_metrics:
            for col in slot_metrics[slot]:
                dwell_metric[f"{col}_s{slot}"] = slot_metrics[slot][col]
        dwell_metrics.append(dwell_metric)

        if len(dwell_metrics) > 1:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            Table(dwell_metrics).write(
                Path(outdir) / "dwell_metrics.csv", overwrite=True
            )

        if len(dwell_events) > 0:
            year = int(CxoTime(start).frac_year)
            event_outdir = (
                Path(outdir) / "events" / f"{year}" / f"dwell_{manvr.kalman_start}"
            )
            make_event_report(start, stop, obsid, dwell_events, event_outdir)
            json.dump(
                dwell_events.as_array().tolist(),
                open((Path(event_outdir) / "events.json"), "w"),
            )
            if len(bgd_events) > 0:
                bgd_events = vstack([Table(bgd_events), dwell_events])
            else:
                bgd_events = dwell_events

    return bgd_events, time_last_processed


def get_outer_min(imgs: np.ndarray, rank: int = 0) -> np.ndarray:
    """
    Get the outer min of the 8x8 image data.

    This function takes a stack of 8x8 images and returns the Nth min of the outer pixels.
    The outer pixels include the 6x6 corner pixels and the 8x8 edge pixels.

    Parameters
    ----------
    imgs : numpy.ndarray
        Array of 8x8 images
    rank : int
        The rank of the min to return

    Returns
    -------
    outer_min : numpy.ndarray
        Array of the Nth min of the outer pixels
    """

    img_len = len(imgs)

    # Make a mask to get just the edges of the 8x8 numpy array
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    # Add the 6x6 corners?
    mask[1, 1] = True
    mask[1, -2] = True
    mask[-2, 1] = True
    mask[-2, -2] = True

    flat_mask = mask.flatten()
    used_pix = np.count_nonzero(flat_mask)
    tile_mask = np.tile(flat_mask, (img_len, 1))

    outer_min = np.sort(
        imgs.reshape(img_len, 64)[tile_mask].reshape(img_len, used_pix), axis=-1
    )[:, rank]

    return outer_min


def get_star_contrib(mag: float) -> tuple:
    """
    Get the star contribution to the background.

    This use an ad-hoc polynomial fit to the star contribution to the background
    and the error on that fit.

    Parameters
    ----------
    mag : float
        The magnitude of the star

    Returns
    -------
    tuple (mag_contrib, mag_contrib_err)
        mag_contrib : float
            The star contribution to the background
        mag_contrib_err : float
            The error on the star contribution to the background
    """
    counts = mag_to_count_rate(mag)
    # mag_p = [2.5e-10, 3.0e-5, 0]
    mag_p = [1.55e-4, 0]
    mag_error_p = [1.822e-05, 2.393]
    mag_contrib = np.clip(np.polyval(mag_p, counts), 0, None)
    mag_contrib_err = np.clip(np.polyval(mag_error_p, counts), 0, None)
    return mag_contrib, mag_contrib_err


def get_background_data_and_thresh(
    slot_data: Table,
    mag: float,
    default_six_threshold: float = None,
    default_eight_threshold: float = None,
) -> dict:
    if default_six_threshold is None:
        default_six_threshold = SIX_THRESHOLD

    if default_eight_threshold is None:
        default_eight_threshold = EIGHT_THRESHOLD

    bgds = {}
    bgds["bgdavg_es"] = slot_data["BGDAVG"] * 5 / 1.696

    ok_8 = (slot_data["IMGSIZE"] == 8) & (slot_data["IMGFUNC"] == 1)

    imgs_8x8_bgsub_dn = slot_data["IMG_BGSUB"]
    imgs_8x8_bgsub = imgs_8x8_bgsub_dn * 5 / 1.696

    outer_min_7 = np.zeros(len(slot_data))
    outer_min_7[ok_8] = get_outer_min(imgs_8x8_bgsub[ok_8], rank=7)
    if np.count_nonzero(ok_8) > 2:
        outer_min_7[~ok_8] = interpolate(
            outer_min_7[ok_8], slot_data["TIME"][ok_8], slot_data["TIME"][~ok_8]
        )
    bgds["outer_min_7"] = outer_min_7

    # Subtract off an estimate of the star or fid contribution by magnitude
    mag_contrib_dn, mag_contrib_err_dn = get_star_contrib(mag)
    mag_contrib = mag_contrib_dn * 5 / 1.696
    mag_contrib_err = mag_contrib_err_dn * 5 / 1.696
    bgds["outer_min_7_magsub"] = np.clip(outer_min_7 - mag_contrib, 0, None)

    # Initialize with a default threshold, but increase threshold for
    # 8x8 data if the star contribution is significant.
    bgds["threshold"] = np.ones_like(outer_min_7) * default_eight_threshold
    bgds["threshold"][ok_8] = np.max([default_eight_threshold, 6 * mag_contrib_err])

    # Set the threshold for 6x6 data if there is any
    ok6 = (slot_data["IMGFUNC"] == 1) & (slot_data["IMGSIZE"] == 6)
    bgds["threshold"][ok6] = default_six_threshold

    bgds["bgd"] = np.where(
        slot_data["IMGSIZE"] == 8,
        bgds["outer_min_7_magsub"],
        bgds["bgdavg_es"],
    )
    return bgds


def get_guide_cat(time: CxoTimeLike) -> Table:
    """
    Get the guide catalog for a given time.

    Parameters
    ----------
    time : str
        Time to get guide catalog

    Returns
    -------
    guide_cat : astropy.table.Table
        Table of guide catalog data
    """
    starcats = get_starcats(CxoTime(time).secs - (3600 * 4), CxoTime(time).secs)
    if len(starcats) == 0:
        LOGGER.info(f"No starcat data for {CxoTime(time).date}")
        return Table()
    starcat = starcats[-1]
    guide_cat = starcat[starcat["type"] != "ACQ"]
    return guide_cat


def get_slot_mags(time: CxoTimeLike) -> dict:
    """
    Get the slot magnitudes for a given time.

    Parameters
    ----------
    time : str
        Time to get slot magnitudes

    Returns
    -------
    slot_mag : dict
        Dictionary of slot magnitudes keyed by slot
    """
    guide_cat = get_guide_cat(time)

    # for any undefined from get_starcats, set to 15 instead of -999.00
    guide_cat[guide_cat["mag"] == -999.00] = 15

    # Initialize the slot magnitudes to 15 (faint enough to not have impact)
    slot_mag = {slot: 15 for slot in range(8)}
    for slot in range(8):
        guide_slots = guide_cat[guide_cat["slot"] == slot]
        if len(guide_slots) != 1:
            continue
        guide_slot = guide_slots[0]
        if np.in1d(guide_slot["type"], ["BOT", "GUI"]):
            # Use the full current agasc for these lookups instead of the commanded magnitude
            star = agasc.get_star(guide_slot["id"], agasc_file="agasc*")
            slot_mag[slot] = star["MAG_ACA"]
        else:
            slot_mag[slot] = guide_slot["mag"]
    return slot_mag


def get_slots_metrics(slots_data: dict) -> dict:
    """
    Get metrics for each slot

    Parameters
    ----------
    slots_data : dict
        A dictionary of slot data tables with keys 0-7 and tables with columns
        "TIME", "IMGFUNC", "IMGSIZE", "outer_min_7_magsub", "bgdavg_es", "threshold"

    Returns
    -------
    slots_metrics : dict
        Dictionary of metrics for each slot
    """
    slots_metrics = {}
    for slot in range(8):
        if slot not in slots_data:
            continue
        slot_data = slots_data[slot]
        ok = slot_data["IMGFUNC"] == 1
        if np.count_nonzero(ok) == 0:
            continue
        if np.median(slot_data["IMGSIZE"][ok]) == 6:
            ok6 = (slot_data["IMGFUNC"] == 1) & (slot_data["IMGSIZE"] == 6)
            slots_metrics[slot] = {
                "bg_col": "bgdavg_es",
                "threshold": np.median(slot_data["threshold"][ok6]),
                "max": slot_data["bgdavg_es"][ok6].max(),
            }
        elif np.median(slot_data["IMGSIZE"][ok]) == 8:
            ok8 = (slot_data["IMGFUNC"] == 1) & (slot_data["IMGSIZE"] == 8)
            slots_metrics[slot] = {
                "bg_col": "outer_min_7_magsub",
                "max": slot_data["outer_min_7_magsub"][ok8].max(),
                "threshold": np.median(slot_data["threshold"][ok8]),
            }
        else:
            continue
    return slots_metrics


def get_manvr_events(start: CxoTimeLike, stop: CxoTimeLike, obsid: int) -> tuple:
    """
    Review a single dwell for high background events.

    Parameters
    ----------
    start : CxoTimeLike
        Start of interval to process
    stop : CxoTimeLike
        End of interval to process
    obsid : int
        Observation ID

    Returns
    -------
    tuple (bgd_events, dwell_stop, slots_metrics)
        bgd_events : astropy.table.Table
            Table of high background events
        dwell_stop : CxoTime
            Stop time of the last processed interval
        slots_metrics : dict
            Dictionary of metrics for each slot

    """
    slot_mag = get_slot_mags(start)

    bgd_events = []

    try:
        sd_table = get_aca_images(start, stop, source="cxc", bgsub=True)
    except Exception:
        LOGGER.info(f"Failed to get image data for dwell {start}")
        return [], None, {}

    slots_data = {slot: sd_table[sd_table["IMGNUM"] == slot] for slot in range(8)}
    for slot, s_data in slots_data.items():
        mag = slot_mag[slot] if slot in slot_mag else 15
        bgds = get_background_data_and_thresh(s_data, mag)
        for key in bgds:
            s_data[key] = bgds[key]

    slots_metrics = get_slots_metrics(slots_data)

    # Check that the image data is complete for the dwell.
    # This assumes that it is sufficient to check slot 3
    if (len(slots_data[3]) == 0) or (
        CxoTime(stop).secs - slots_data[3]["TIME"][-1]
    ) > 60:
        LOGGER.info(f"Stopping review of dwells at dwell {start}, missing image data")
        return [], None, {}

    raw_events = get_raw_events(slots_data)

    merged_events = combine_events(raw_events)

    for event in merged_events:
        event["obsid"] = obsid
        event["dwell_datestart"] = start
        event["dwell_datestop"] = CxoTime(stop).date
        event["datestart"] = CxoTime(event["tstart"]).date
        stats = get_event_stats(event, slots_data)
        for key in stats:
            event[key] = stats[key]
        bgd_events.append(event)

    if len(bgd_events) > 0:
        dwell_extra_data = get_manvr_extra_data(start, stop)
        for e in bgd_events:
            for key in dwell_extra_data:
                e[key] = dwell_extra_data[key]
        bgd_events = Table(bgd_events)

    return bgd_events, stop, slots_metrics


def make_event_report(
    dwell_start: CxoTimeLike,
    dwell_stop: CxoTimeLike,
    obsid: int,
    obs_events: Table,
    outdir: Path = ".",
    redo: bool = True,
) -> None:
    """
    Make a report for a high background event.

    Parameters
    ----------
    dwell_start : CxoTimeLike
        Start of dwell
    dwell_stop : CxoTimeLike
        End of dwell
    obsid : int
        Observation ID
    obs_events : astropy.table.Table
        Table of high background events
    outdir : Path
        Output directory
    redo : bool
        Redo the report if it already exists
    """
    if not Path(outdir).exists():
        # Make the directory if it doesn't exist using Path
        Path(outdir).mkdir(parents=True, exist_ok=True)
    elif redo is False:
        return

    # Make a plot of the top 5 events
    obs_events.sort("max_bgd", reverse=True)
    events_limit_5 = obs_events[:5].copy()
    events_limit_5.sort("tstart")

    events = []
    for index, e in enumerate(events_limit_5):
        event = dict(zip(e.colnames, e.as_void(), strict=False))
        event["img_html"] = plot_images(event["tstart"] - 10, event["tstart"] + 100)
        event["index"] = index
        events.append(event)

    # Make a plot of the full dwell
    bgd_html = plot_dwell(dwell_start, dwell_stop, dwell_start, events)

    LOGGER.info(f"Making report for {outdir}")
    file_dir = Path(__file__).parent
    obs_template = Template(open(file_dir / "per_obs_template.html", "r").read())
    page = obs_template.render(
        dwell_datestart=events[0]["dwell_datestart"],
        dwell_datestop=events[0]["dwell_datestop"],
        obsid=obsid,
        events=events,
        bgd_html=bgd_html,
    )
    f = open(Path(outdir) / "index.html", "w")
    f.write(page)
    f.close()


def rebin_data(
    times: np.ndarray, data: np.ndarray, num_bins: int, func: Callable
) -> tuple:
    """
    Rebin data to a fixed number of bins.

    Parameters
    ----------
    times : numpy.ndarray
        Array of times
    data : numpy.ndarray
        Array of data
    num_bins : int
        Number of bins to rebin to
    func : callable
        Function to use to aggregate the data in each bin (examples: np.mean, np.max)

    Returns
    -------
    tuple (binned_times, binned_data)
        binned_times : numpy.ndarray
            Array of binned times
        binned_data : numpy.ndarray
            Array of binned data
    """
    # Calculate the bin edges
    bin_edges = np.linspace(times.min(), times.max(), num_bins + 1)

    # Digitize the data to find out which bin each point belongs to
    bin_indices = np.digitize(times, bin_edges)

    # Initialize arrays to store the binned data
    binned_times = np.zeros(num_bins)
    binned_data = np.zeros(num_bins)

    # Calculate the mean of the data points within each bin
    for i in range(1, num_bins + 1):
        ok = bin_indices == i
        if np.any(ok):
            binned_times[i - 1] = np.mean(times[ok])
            binned_data[i - 1] = func(data[ok])

    # Filter out bins that have no data points
    valid_bins = binned_times != 0
    binned_times = binned_times[valid_bins]
    binned_data = binned_data[valid_bins]
    return binned_times, binned_data


def plot_dwell(
    start: CxoTimeLike, stop: CxoTimeLike, dwell_start: CxoTimeLike, events: Table
) -> str:
    """
    Generate a plotly plot of backgrounds data and aokalstr over an event.

    This function creates a plot with three subplots:
    1. BGDAVG (e-/s)
    2. Corrected background 8th outer min (e-/s)
    3. AOKALSTR (N)

    The function bins the data to reduce the number of points for faster plotting.

    Parameters
    ----------
    start : CxoTimeLike
        Start time of the plotting range.
    stop : CxoTimeLike
        End time of the plotting range.
    dwell_start : CxoTimeLike
        Start time of the dwell.
    events : Table
        Table of events to be plotted.

    Returns
    -------
    str
        HTML representation of the plot.

    """
    slot_mag = get_slot_mags(dwell_start)
    sd_table = get_aca_images(start, stop, source="cxc", bgsub=True)
    slots_data = {slot: sd_table[sd_table["IMGNUM"] == slot] for slot in range(8)}

    # Define a color palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("BGDAVG (e-/s)", "corr background 8th outer min (e-/s)"),
    )

    # use a fixed number of bins to speed up the plotting if there are a lot of data points
    num_bins = 2000

    for slot in range(8):
        slot_data = slots_data[slot]

        if len(slot_data["TIME"]) == 0:
            raise ValueError
        mag = slot_mag[slot] if slot in slot_mag else 15
        bgds = get_background_data_and_thresh(slot_data, mag)
        for key in bgds:
            slot_data[key] = bgds[key]
        ok = slot_data["IMGFUNC"] == 1

        if np.count_nonzero(ok) == 0:
            continue

        times = slot_data["TIME"][ok] - CxoTime(start).secs
        dtimes = (times - times[0]) / 1000.0
        y1_data = slot_data["bgdavg_es"][ok]
        y2_data = slot_data["outer_min_7_magsub"][ok]

        # Bin the data to make these plots faster
        if len(dtimes) > num_bins:
            a_times, a_data = rebin_data(dtimes, y1_data, num_bins, np.max)
            _, b_data = rebin_data(dtimes, y2_data, num_bins, np.max)
            dtimes = a_times
            y1_data = a_data
            y2_data = b_data

        fig.add_trace(
            go.Scatter(
                x=dtimes,
                y=y1_data,
                mode="markers",
                name=f"Slot {slot}",
                marker={"color": colors[slot]},
                legendgroup=f"Slot {slot}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dtimes,
                y=y2_data,
                mode="markers",
                marker={"color": colors[slot]},
                legendgroup=f"Slot {slot}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        # Add horizontal line for the threshold in a subplot
        threshold = np.median(slot_data["threshold"][ok])
        if np.median(slot_data["IMGSIZE"][ok]) == 6:
            figure_col = 1
        elif np.median(slot_data["IMGSIZE"][ok]) == 8:
            figure_col = 2
        else:
            continue
        fig.add_trace(
            go.Scatter(
                x=[min(dtimes), max(dtimes)],
                y=[threshold, threshold],
                mode="lines",
                line={"color": colors[slot], "width": 2, "dash": "dash"},
                legendgroup=f"Slot {slot}",
                showlegend=False,
            ),
            row=1,
            col=figure_col,
        )

    # Add the range of the events to the first two plots as a shaded region
    for event in events:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=(event["tstart"] - CxoTime(start).secs) / 1000.0,
            x1=(event["tstop"] - CxoTime(start).secs) / 1000.0,
            y0=0,
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
        fig.add_shape(
            type="rect",
            xref="x2",
            yref="paper",
            x0=(event["tstart"] - CxoTime(start).secs) / 1000.0,
            x1=(event["tstop"] - CxoTime(start).secs) / 1000.0,
            y0=0,
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

    aokalstr = fetch.Msid("AOKALSTR", CxoTime(start).secs, CxoTime(stop).secs)
    values = np.array(aokalstr.vals).astype(int)
    dtimes = (aokalstr.times - aokalstr.times[0]) / 1000.0
    if len(dtimes) > num_bins:
        a_times, a_data = rebin_data(dtimes, values, num_bins, np.min)
        dtimes = a_times
        values = a_data

    fig.add_trace(
        go.Scatter(x=dtimes, y=values, mode="lines", name="aokalstr"),
        row=1,
        col=3,
    )

    fig.update_xaxes(matches="x")  # Define the layout
    fig.update_layout(
        xaxis_title="Obs Time (ks)",
        yaxis_title="BGDAVG (e-/s)",
        xaxis2_title="Obs Time (ks)",
        yaxis2_title="corr background 8th outer min",
        xaxis3_title="Obs Time (ks)",
        yaxis3_title="AOKALSTR (N)",
        yaxis={"range": [0, 3000]},
        yaxis2={"range": [0, 500]},
        yaxis3={"range": [0, 8]},
        height=400,
        width=1200,
        legend={
            "font": {"size": 10},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.2,
            "yanchor": "top",
        },
    )

    # Convert the figure to HTML
    html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True},
    )

    return html


def get_images_for_plot(start: CxoTimeLike, stop: CxoTimeLike) -> tuple:
    """
    Retrieve and process ACA images for plotting within a specified time range.

    This function retrieves ACA images from the specified start to stop times, processes
    the images to extract background data and thresholds, and organizes the data into
    stacks for each slot. It also calculates background averages and outer minimum values
    for each slot.

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the image retrieval.
    stop : CxoTimeLike
        Stop time for the image retrieval.

    Returns
    -------
    tuple (times, image_stacks, bgdavgs, outer_mins)
        image_stacks (list): A list of image stacks for each slot.
        bgdavgs (dict): A dictionary with slot numbers as keys and lists of background averages.
        outer_mins (dict): A dictionary with slot numbers as keys and lists of outer minimum values.
        times (numpy.ndarray): An array of unique times within the specified range.

    """
    slot_mag = get_slot_mags(start.secs)

    sd_table = get_aca_images(start.secs - 10, stop.secs + 10, source="cxc", bgsub=True)
    slots_data = {slot: sd_table[sd_table["IMGNUM"] == slot] for slot in range(8)}

    for slot in range(8):
        mag = slot_mag[slot] if slot in slot_mag else 15
        slots_data[slot] = slots_data[slot]
        slots_data[slot]["SLOT"] = slot
        bgds = get_background_data_and_thresh(slots_data[slot], mag)
        for key in bgds:
            slots_data[slot][key] = bgds[key]

    # Get a list of all the unique times in the set
    times = np.unique(sd_table["TIME"])
    times = times[(times >= (start.secs - 4.5)) & (times <= (stop.secs + 4.5))]

    image_mask_fill = 0
    image_stacks = []
    bgdavgs = collections.defaultdict(list)
    outer_mins = collections.defaultdict(list)

    for slot in range(8):
        slot_data = slots_data[slot]
        slot_stack = []
        for time in times:
            sz = 8
            # Find last complete row for this slot at or before this time
            ok = slot_data["TIME"] <= time
            if np.any(ok):
                idx = np.flatnonzero(ok)[-1]
                bgdavgs[slot].append(slot_data["bgdavg_es"][idx])
                outer_mins[slot].append(slot_data["outer_min_7_magsub"][idx])
                img = slot_data["IMG"][idx].reshape(8, 8)
                # if this is a 6x6 image, explicitly fill the edges with something
                # not as annoying as the large value
                if np.count_nonzero(img.mask) == 28:
                    img.fill_value = image_mask_fill
                    img = img.filled()

                img = np.clip(img, a_min=10, a_max=None)
                if np.all(img == 10) or slot_data["IMGFUNC"][idx] != 1:
                    pixvals = np.zeros((sz, sz))
                    np.fill_diagonal(pixvals, 255)
                else:
                    logimg = np.log(img)
                    pixvals = 255 * (logimg - 4) / (np.max(logimg) - 4)
            else:
                outer_mins[slot].append(0)
                bgdavgs[slot].append(0)
                pixvals = np.zeros((sz, sz))
                np.fill_diagonal(pixvals, 255)

            img = np.zeros((10, 10))
            img[1:9, 1:9] = pixvals.transpose()
            slot_stack.append(img)
        image_stacks.append(slot_stack)
    return times, image_stacks, bgdavgs, outer_mins


def plot_images(start: CxoTimeLike, stop: CxoTimeLike) -> str:  # noqa: PLR0915 Too many statements
    """
    Create an animated plot of ACA images over time.

    This function generates a grid of heatmap images for the given times and pixel values,
    and creates an animated plot with play and pause controls. The images are organized
    into stacks, and each stack is displayed in a separate subplot.

    Parameters
    ----------
    start = CxoTimeLike
        Start time for the image retrieval.
    stop = CxoTimeLike
        Stop time for the image retrieval.

    Returns
    -------
    str
        HTML representation of the animated plot.

    """
    start = CxoTime(start)
    stop = CxoTime(stop)

    times, image_stacks, bgdavgs, outer_mins = get_images_for_plot(start, stop)

    # Parameters
    num_stacks = len(image_stacks)
    num_frames = len(times)
    COLORMAP = "inferno"

    # Create subplot grid (1 row and 8 columns)
    fig = make_subplots(
        rows=1,
        cols=num_stacks,
        horizontal_spacing=0.02,  # Adjust spacing between plots
    )

    # Add initial traces for the first frame using `Heatmap`
    for idx, stack in enumerate(image_stacks):
        fig.add_trace(
            go.Heatmap(
                z=stack[0],
                colorscale=COLORMAP,
                showscale=False,  # Shared colorbar
                zmin=0,  # Explicit color range
                zmax=255,  # Explicit color range
            ),
            row=1,
            col=idx + 1,
        )

    # Update layout with shared colorbar and axis properties
    fig.update_layout(
        height=300,
        width=1200,
        margin={"t": 40, "b": 20},
        coloraxis={
            "colorscale": COLORMAP,
            "cmin": 0,
            "cmax": 255,
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
                "x": 0,
                "xanchor": "left",
                "y": -1.2,
                "yanchor": "bottom",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0, "easing": "linear"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [f"frame_{i}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": f"{i}",
                    }
                    for i in range(num_frames)
                ],
            }
        ],
    )

    # Hide axis tick labels, grids, and zero lines for all subplots
    for i in range(1, num_stacks + 1):
        fig.update_layout(
            {
                f"xaxis{i}": {
                    "showticklabels": False,
                    "showgrid": False,
                    "zeroline": False,
                    "ticks": "",
                    "scaleanchor": f"y{i}",  # Enforce square scaling for each subplot
                    "constrain": "domain",  # Ensures no extra padding
                },
                f"yaxis{i}": {
                    "showticklabels": False,
                    "showgrid": False,
                    "zeroline": False,
                    "ticks": "",
                    "constrain": "domain",  # Ensures no extra padding
                },
            }
        )

    # Calculate the exact center for each subplot using domain properties
    subplot_centers = [
        (fig.layout[f"xaxis{i}"].domain[0] + fig.layout[f"xaxis{i}"].domain[1]) / 2
        for i in range(1, num_stacks + 1)
    ]

    # Adjust the y-position of the labels
    label_y_position = -0.4  # Move closer to the images

    # Create frames for the animation
    frames = []
    for frame_idx in range(num_frames):
        frame_data = [
            go.Heatmap(
                z=stack[frame_idx],
                colorscale=COLORMAP,
                showscale=False,  # Shared colorbar
                zmin=0,
                zmax=255,
            )
            for stack in image_stacks
        ]

        # Centered labels: use calculated subplot centers
        new_titles = [
            f"bgdavg: {bgdavgs[i][frame_idx]:.0f}<br>outermin: {outer_mins[i][frame_idx]:.0f}"
            for i in range(num_stacks)
        ]
        annotations = [
            {
                "x": subplot_centers[stack_idx],  # Use exact center of each subplot
                "y": label_y_position,  # Adjusted position closer to images
                "xref": "x domain" if stack_idx == 0 else f"x{stack_idx + 1} domain",
                "yref": "y domain" if stack_idx == 0 else f"y{stack_idx + 1} domain",
                "text": new_titles[stack_idx],
                "showarrow": False,
                "font": {"size": 12},
                "align": "left",  # Center-align text
            }
            for stack_idx in range(num_stacks)
        ]
        frames.append(
            go.Frame(
                data=frame_data,
                name=f"frame_{frame_idx}",
                layout=go.Layout(
                    title=f"Date: {CxoTime(times[frame_idx]).date}",
                    annotations=annotations,
                ),
            )
        )

    # Add frames to the figure
    fig.frames = frames

    # fig.show()
    # filename = "images_{}.html".format(CxoTime(start).date)
    html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True},
        auto_play=False,
    )

    return html


def make_summary_reports(bgd_events, outdir="."):
    """
    Make event reports

    For a table of background events, make a high level summary report
    by obsid and make an individual obsid report for each.

    Parameters
    ----------
    bgd_events : astropy table
        Table of background events
    outdir : str
        Output directory

    Returns
    -------
    None
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Get notes
    extra_notes = get_extra_notes()

    # For the HTML page, reduce to just the significant events
    ok = significant_events(bgd_events)
    bgd_events = bgd_events[ok]

    dwell_events = []
    for dwell_start in np.unique(bgd_events["dwell_datestart"]):
        events = bgd_events[bgd_events["dwell_datestart"] == dwell_start]

        # ignore the obsid == 0 case(s) for now
        if np.all(events["obsid"] == 0):
            continue

        slots = {}
        for e in events:
            for slot in range(8):
                if e[f"s{slot}"] == 1:
                    slots[slot] = 1

        # Save the events, the intervals, and some other useful stuff for the per-dwell table
        obs = {
            "events": events.copy(),
            "first_event_start": CxoTime(events[0]["tstart"]).date,
            "dwell_datestart": dwell_start,
            "n_events": len(events),
            "max_dur": np.max(events["duration"]),
            "max_slot_secs": np.max(events["slot_seconds"]),
            "n_slots": len(slots),
            "obsid": events["obsid"][0],
            "reldir": f"events/{int(CxoTime(dwell_start).frac_year)}/dwell_{dwell_start}",
            "dir": Path(outdir)
            / "events"
            / f"{int(CxoTime(dwell_start).frac_year)}"
            / f"dwell_{dwell_start}",
            "pitch": events[0]["pitch"],
            "notes": events[0]["notes"],
            "manual_notes": " " * 25,
        }

        # Add extra notes
        if dwell_start in extra_notes["dwell_datestart"]:
            ok = extra_notes["dwell_datestart"] == dwell_start
            obs["manual_notes"] = extra_notes["notes"][ok][0]

        dwell_events.append(obs)

    if len(dwell_events) > 0:
        events_top_html = plot_events_top(dwell_events)
        events_by_pitch_html = plot_events_pitch(bgd_events)
    else:
        events_top_html = ""
        events_by_pitch_html = ""

    dwell_events = sorted(dwell_events, key=lambda i: i["dwell_datestart"])
    dwell_events = dwell_events[::-1]

    file_dir = Path(__file__).parent
    template = Template(open(file_dir / "top_level_template.html", "r").read())
    page = template.render(
        obs_events=dwell_events,
        events_top_html=events_top_html,
        events_by_pitch_html=events_by_pitch_html,
    )
    f = open(Path(outdir) / "index.html", "w")
    f.write(page)
    f.close()


def plot_events_top(dwell_events):
    """
    Make a bar chart of the number of events per year.

    Parameters
    ----------
    dwell_events : astropy table
        Table of high background events

    Returns
    -------
    html : str
        HTML representation
    """
    fig = go.Figure()

    # Count the events in each calendar quarter and make a bar chart
    frac_years = np.array(
        [CxoTime(d["dwell_datestart"]).frac_year for d in dwell_events]
    )

    # bin the data by year
    bins = np.arange(np.floor(frac_years.min()), frac_years.max() + 0.25, 1.0)

    hist, _ = np.histogram(frac_years, bins=bins)
    years = [f"{int(b):04d}" for b in bins[:-1]]

    fig.add_trace(go.Bar(x=years, y=hist, name="Events"))
    fig.update_layout(
        title="Number of ACA HI BGD events per year",
        xaxis_title="Year",
        yaxis_title="Number of events",
        height=400,
        width=600,
    )

    fig.update_xaxes(tickangle=-90)

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True},
    )


def plot_events_pitch(dwell_events: Table) -> str:
    """
    Make a plotly scatter plot of the pitch of high background events over time.

    Parameters
    ----------
    dwell_events : astropy table
        Table of high background events

    Returns
    -------
    html : str
        HTML representation of the plot
    """
    fig = go.Figure()

    # Get the pitch values and the dates
    pitches = [d["pitch"] for d in dwell_events]
    frac_year = [CxoTime(d["dwell_datestart"]).frac_year for d in dwell_events]

    fig.add_trace(go.Scatter(x=frac_year, y=pitches, mode="markers"))

    fig.update_layout(
        title="Pitch of ACA HI BGD events over time",
        xaxis_title="Time",
        yaxis_title="Pitch",
        height=400,
        width=600,
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True},
    )


def significant_events(bg_events: Table) -> np.array:
    """
    Filter out events that are not significant.

    This function filters out events that have obsid == 0 or obsid == -1, and
    have either less than n_slots or less than slot_seconds.

    Parameters
    ----------
    bg_events : astropy table
        Table of background events

    Returns
    -------
    ok : np.array
        Boolean array of significant events
    """
    # Filter not null obsid, and either >= n_slots  or >= than duration seconds
    ok = ~np.in1d(bg_events["obsid"], [0, -1]) & (
        (bg_events["n_slots"] >= 5)
        | (bg_events["slot_seconds"] >= 60)
        | (bg_events["notes"] != "")
    )
    return ok


def review_and_send_email(events, opt):
    """
    Review the events and send email if needed.

    Parameters
    ----------
    events : astropy table
        Table of events
    opt : argparse.Namespace
        Command line options

    Returns
    -------
    None
    """
    for dwell_start in np.unique(events["dwell_datestart"]):
        obs_events = events[events["dwell_datestart"] == dwell_start]
        obsid = obs_events["obsid"][0]
        year = int(CxoTime(dwell_start).frac_year)
        url = f"{opt.web_url}/events/{year}/dwell_{dwell_start}"
        send_mail(
            LOGGER,
            opt,
            f"ACA HI BGD event in obsid {obsid}",
            f"HI BGD in obsid {obsid} report at {url}",
            __file__,
        )


def main(args=None):  # noqa: PLR0912, PLR0915 too many branches, too many statements
    """
    Do high background processing.

    Review dwells (taken from kadi maneuvers) for new high background events, update a text
    file table of those events, make reports, and notify via email as needed.
    """

    opt = get_opt().parse_args(args)
    log_run_info(LOGGER.info, opt, version=__version__)

    EVENT_ARCHIVE = Path(opt.data_root) / "bgd_events.dat"
    Path(opt.data_root).mkdir(parents=True, exist_ok=True)
    start = None

    bgd_events = []
    if Path(EVENT_ARCHIVE).exists():
        bgd_events = Table.read(EVENT_ARCHIVE, format="ascii.ecsv")
    if len(bgd_events) > 0:
        start = CxoTime(bgd_events["dwell_datestart"][-1])
        # Remove any bogus events from the real list
        bgd_events = bgd_events[bgd_events["obsid"] != -1]

        bads = get_false_positives()
        # Remove any known bad events from the real list too
        for bad_datestart in bads["dwell_datestart"]:
            bgd_events = bgd_events[bgd_events["dwell_datestart"] != bad_datestart]

        # Fill masked values in the notes column with empty strings
        bgd_events["notes"] = bgd_events["notes"].filled("")

    if opt.replot:
        for dwell_start in np.unique(bgd_events["dwell_datestart"]):
            obs_events = bgd_events[bgd_events["dwell_datestart"] == dwell_start]
            obsid = obs_events["obsid"][0]
            year = int(CxoTime(dwell_start).frac_year)
            url = f"{opt.web_url}/events/{year}/dwell_{dwell_start}"
            LOGGER.info(f"Replotting HI BGD event in obsid {obsid} {url}")
            make_event_report(
                dwell_start,
                dwell_start,
                obsid,
                obs_events,
                outdir=Path(opt.web_out)
                / "events"
                / f"{year}"
                / f"dwell_{dwell_start}",
                redo=True,
            )
        make_summary_reports(bgd_events, outdir=opt.web_out)
        return

    # If the user has asked for a start time earlier than the end of the
    # table, delete possibly conflicting events in the table.
    if opt.start is not None and opt.stop is not None:
        if start is not None:
            bgd_events = bgd_events[
                (bgd_events["dwell_datestart"] < CxoTime(opt.start).date)
                | (bgd_events["dwell_datestart"] > CxoTime(opt.stop).date)
            ]
        start = CxoTime(opt.start)
    if start is None:
        start = CxoTime(-7)

    new_events, last_proc_time = get_events(start, stop=opt.stop, outdir=opt.web_out)
    if len(new_events) > 0:
        new_events = Table(new_events)

        ok = significant_events(new_events)
        big_events = new_events[ok]
        if len(opt.emails) > 0:
            review_and_send_email(events=big_events, opt=opt)

        ok = ~np.in1d(new_events["obsid"], [0, -1])
        for dwell_start in np.unique(new_events["dwell_datestart"][ok]):
            obs_events = new_events[new_events["dwell_datestart"] == dwell_start]
            year = int(CxoTime(dwell_start).frac_year)
            obsid = obs_events["obsid"][0]
            url = f"{opt.web_url}/events/{year}/dwell_{dwell_start}"
            LOGGER.warning(f"HI BGD event in obsid {obsid} {url}")

    if len(bgd_events) > 0:
        bgd_events = vstack([bgd_events, new_events])
    else:
        bgd_events = new_events

    if len(bgd_events) == 0:
        LOGGER.warning("No new or old events - Bailing out")
        return

    bgd_events.sort("datestart")

    # Add a null event at the end
    bgd_events.add_row()
    bgd_events[-1]["obsid"] = -1
    bgd_events[-1]["dwell_datestart"] = CxoTime(last_proc_time).date

    bgd_events.write(EVENT_ARCHIVE, format="ascii.ecsv", overwrite=True)

    make_summary_reports(bgd_events, outdir=opt.web_out)


if __name__ == "__main__":
    main()
