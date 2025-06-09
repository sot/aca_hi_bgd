import argparse
import functools
from collections.abc import Callable
from pathlib import Path

import agasc
import numpy as np
from acdc.common import send_mail
from astropy.table import Table, vstack
from chandra_aca.aca_image import get_aca_images
from chandra_aca.transform import mag_to_count_rate
from cheta import fetch
from cxotime import CxoTime, CxoTimeLike
from jinja2 import Template
from kadi import events
from kadi.commands import get_starcats
from ska_helpers.logging import basic_logger
from ska_helpers.run_info import log_run_info
from ska_numpy import interpolate

from aca_hi_bgd import __version__
from aca_hi_bgd.plots import (
    plot_dwell,
    plot_events_delay,
    plot_events_pitch,
    plot_events_rel_perigee,
    plot_events_top,
    plot_images,
)

DETECT_HITS = 3
DETECT_WINDOW = 21  # seconds
SIX_THRESHOLD = 580  # applied to BGDAVG scaled to e-/s
EIGHT_THRESHOLD = 140  # applied to 8th outer min scaled to e-/s

DOC_ID = "1GoYBTIQAv0qq2vh3jYxHBYHfEq2I8LVGMiScDX7OFvw"
GID = 524798125
url_start = "https://docs.google.com/spreadsheets/d"
GSHEET_URL = f"{url_start}/{DOC_ID}/export?format=csv&id={DOC_ID}&gid={GID}"
GSHEET_USER_URL = f"{url_start}/{DOC_ID}/edit?usp=sharing"


LOGGER = basic_logger(__name__, level="INFO")
fetch.data_source.set("cxc", "maude allow_subset=False")


def get_opt():
    parser = argparse.ArgumentParser(description="High Background event finder")
    parser.add_argument("--start", help="Start date")
    parser.add_argument("--stop", help="Stop date")
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Output data directory",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Replot known events",
    )
    parser.add_argument(
        "--web-out",
        default="./webout",
    )
    parser.add_argument(
        "--web-url", default="https://cxc.harvard.edu/mta/ASPECT/aca_hi_bgd_mon"
    )
    parser.add_argument(
        "--email",
        action="append",
        dest="emails",
        default=[],
        help="Email address for notificaion",
    )
    return parser


@functools.cache
def get_hi_bgd_notes(data_root) -> Table:
    """
    Get the high background notes from the Google Sheet.

    Parameters
    ----------
    data_root : str
        The root directory for the data
    Returns
    -------
    dat : astropy.table.Table
        Table of notes and false positives
    """
    LOGGER.info(f"Reading google sheet {GSHEET_URL}")
    dat = None
    try:
        dat = Table.read(GSHEET_URL, format="ascii.csv")
    except Exception as e:
        LOGGER.error(f"Failed to read {GSHEET_URL} with error: {e}")

    if dat is not None:
        dat.write(
            Path(data_root) / "hi_bgd_notes.csv",
            format="ascii.csv",
            overwrite=True,
        )
    else:
        dat = Table.read(Path(data_root) / "hi_bgd_notes.csv", format="ascii.csv")

    return dat


def get_false_positives(data_root) -> Table:
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
    dat = get_hi_bgd_notes(data_root)
    ok = dat["type"] == "false positive"
    return dat[ok]


def get_extra_notes(data_root) -> Table:
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
    dat = get_hi_bgd_notes(data_root)
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
    times_to_check = list(np.unique(hit_times))

    # Add one more time to check that should not have any counts
    # This is needed to reliably end the intervals.
    times_to_check.append(1.025)

    event_start = None
    last_check_time = None
    for time in times_to_check:
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


def combine_events(events: list, tol: float = 90) -> list:
    """
    Combine overlapping event intervals.

    This function takes a list of event intervals and returns a reduced list where overlapping or
    close to overlapping intervals are merged.  The tolerance for merging is defined by the tol
    parameter.

    Parameters
    ----------
    events : list
        List of dictionaries with keys "tstart" and "tstop" for each event.
        Assumed to already be sorted by "tstart"
    tol : float
        The tolerance for merging overlapping events in seconds.

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
            last_event["tstop"] = max(event["tstop"], last_event["tstop"])
        else:
            merged_events.append(event)
    return merged_events


def calculate_slot_seconds(imgsize: np.ndarray) -> float:
    """
    Calculate a metric that adds up the time for each slot where the background is high.

    While the time between images (based on image size) is not really related to the time of the
    high background hit, it is not unreasonable to use the time between images as a "chunk" that
    can be summed to get a metric of how long the background is high.

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

    Parameters
    ----------
    start : CxoTimeLike
        Start of dwell
    stop : CxoTimeLike
        End of dwell

    Returns
    -------
    extra_data : dict
        Dictionary of extra data for the dwell including "pitch" and "notes"
    """

    pitchs = fetch.Msid("DP_PITCH", start, stop)
    if len(pitchs.vals) > 0:
        pitch = np.median(pitchs.vals)
    else:
        pitch = -999

    expected_acqs = 0
    guide_cat = get_guide_cat(start)
    if len(guide_cat) > 0 and "MON" in guide_cat["type"]:
        expected_acqs = 1

    # Check for multiple acquisitions and add notes
    # An extra acquisition is expected for monitor window catalogs
    notes = []
    aoacaseq = fetch.Msid("AOACASEQ", start, stop)
    from cheta.utils import state_intervals

    aca_states = state_intervals(aoacaseq.times, aoacaseq.vals)
    if np.count_nonzero(aca_states["val"] == "AQXN") > expected_acqs:
        notes.append("Full REACQ")

    # Check for NSUN and BRIT
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
    start: CxoTimeLike,
    stop: CxoTimeLike = None,
    outdir: Path = ".",
    data_root: Path = ".",
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
    LOGGER.info(f"Processing with start: {start} and stop: {stop}")
    manvrs = events.manvrs.filter(
        kalman_start__gte=start.date, kalman_start__lte=stop.date, n_dwell__gt=0
    )
    LOGGER.info(f"Found {len(manvrs)} maneuvers to process")

    bgd_events = []
    if len(manvrs) == 0:
        return bgd_events, None

    dwell_metrics = get_previous_events(Path(outdir) / "dwell_metrics.csv")
    dwell_starts = [d["dwell"] for d in dwell_metrics]
    time_last_processed = None

    bads = get_false_positives(data_root)

    for manvr in manvrs:
        try:
            obsid = manvr.obsid
        except ValueError:
            LOGGER.info(f"Skipping manvr {manvr}.")
            continue

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

        dwell_events, dwell_end_time, slot_metrics = get_manvr_events(
            start, stop, obsid
        )

        if dwell_end_time is None:
            continue

        time_last_processed = dwell_end_time

        # Assemble a row of data for the dwell metrics
        dwell_metric = {"dwell": start, "obsid": obsid}
        for slot in slot_metrics:
            for col in slot_metrics[slot]:
                dwell_metric[f"{col}_s{slot}"] = slot_metrics[slot][col]
        dwell_metrics.append(dwell_metric)

        if len(dwell_metrics) > 0:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            Table(dwell_metrics).write(
                Path(outdir) / "dwell_metrics.csv", overwrite=True
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
    """
    Get the corrected background data and thresholds for a given slot.

    Parameters
    ----------
    slot_data : astropy.table.Table
        Table of slot data with columns "TIME", "IMGFUNC", "IMGSIZE", "IMG_BGSUB"
        and "BGDAVG"
    mag : float
        The magnitude of the star
    default_six_threshold : float
        Default threshold for 6x6 data
    default_eight_threshold : float
        Default threshold for 8x8 data

    Returns
    -------
    bgds : dict
        Dictionary of background data and thresholds with keys "bgdavg_es", "outer_min_7",
        "outer_min_7_magsub", "threshold", "bgd"
    """
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
    if len(guide_cat) > 0:
        guide_cat[guide_cat["mag"] == -999.00] = 15

    # Initialize the slot magnitudes to 15 (faint enough to not have impact)
    slot_mag = dict.fromkeys(range(8), 15)
    for slot in range(8):
        if len(guide_cat) == 0:
            continue
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
        LOGGER.info(f"Cannot process dwell {start}, missing image data")
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


def make_dwell_report(
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

    top_events = []
    for index, e in enumerate(events_limit_5):
        event = dict(zip(e.colnames, e.as_void(), strict=False))
        event["img_html"] = plot_images(event["tstart"], event["tstop"])
        event["index"] = index
        top_events.append(event)

    # Make a plot of the full dwell
    bgd_html = plot_dwell(
        dwell_start,
        dwell_stop,
        dwell_start,
        top_events=top_events,
        all_events=obs_events,
    )

    LOGGER.info(f"Making report for {outdir}")
    file_dir = Path(__file__).parent
    obs_template = Template(open(file_dir / "per_obs_template.html", "r").read())
    page = obs_template.render(
        dwell_datestart=top_events[0]["dwell_datestart"],
        dwell_datestop=top_events[0]["dwell_datestop"],
        obsid=obsid,
        events=top_events,
        bgd_html=bgd_html,
        DETECT_HITS=DETECT_HITS,
        DETECT_WINDOW=DETECT_WINDOW,
        SIX_THRESHOLD=SIX_THRESHOLD,
        EIGHT_THRESHOLD=EIGHT_THRESHOLD,
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


def make_summary_reports(bgd_events, outdir=".", data_root="."):
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
    extra_notes = get_extra_notes(data_root)

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
            "max_bgd": np.max(events["max_bgd"]),
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
        events_by_pitch_html = plot_events_pitch(dwell_events)
        events_by_delay_html = plot_events_delay(dwell_events)
        events_rel_perigee_html = plot_events_rel_perigee(bgd_events)
        # events_manvr_html = plot_events_manvr(bgd_events)
    else:
        events_top_html = ""
        events_by_pitch_html = ""
        events_by_delay_html = ""
        events_rel_perigee_html = ""
        # events_manvr_html = ""

    dwell_events = sorted(dwell_events, key=lambda i: i["dwell_datestart"])
    dwell_events = dwell_events[::-1]

    file_dir = Path(__file__).parent
    template = Template(open(file_dir / "top_level_template.html", "r").read())
    page = template.render(
        obs_events=dwell_events,
        events_top_html=events_top_html,
        events_by_pitch_html=events_by_pitch_html,
        events_by_delay_html=events_by_delay_html,
        events_rel_perigee_html=events_rel_perigee_html,
        GSHEET_USER_URL=GSHEET_USER_URL,
    )
    f = open(Path(outdir) / "index.html", "w")
    f.write(page)
    f.close()


def significant_events(bg_events: Table) -> np.array:
    """
    Filter out events that are not significant.

    This function filters out events that have obsid == 0 or obsid == -1, and
    have 5 more more impacted slots or greater than or equal to 60 "slot seconds".
    Dwells with notes are also included in the full list even if they don't
    exceed those thresholds.

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
    # or notes not an empty or whitespace string
    bg_notes = np.array([note.strip() for note in bg_events["notes"]])
    ok = ~np.in1d(bg_events["obsid"], [0, -1]) & (
        (bg_events["n_slots"] >= 5)
        | (bg_events["slot_seconds"] >= 60)
        | (bg_notes != "")
    )
    return ok


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
        # Remove any bogus events from the list
        bgd_events = bgd_events[bgd_events["obsid"] != -1]

        bads = get_false_positives(opt.data_root)
        # Remove any known bad events from the list
        for bad_datestart in bads["dwell_datestart"]:
            bgd_events = bgd_events[bgd_events["dwell_datestart"] != bad_datestart]

        # Fill masked values in the notes column with empty strings
        bgd_events["notes"] = bgd_events["notes"].filled("")

    if opt.replot:
        ok = significant_events(bgd_events)
        for dwell_start in np.unique(bgd_events["dwell_datestart"][ok]):
            obs_events = bgd_events[bgd_events["dwell_datestart"] == dwell_start]
            dwell_stop = obs_events["dwell_datestop"][0]
            obsid = obs_events["obsid"][0]
            year = int(CxoTime(dwell_start).frac_year)
            url = f"{opt.web_url}/events/{year}/dwell_{dwell_start}"
            LOGGER.info(f"Replotting HI BGD event in obsid {obsid} {url}")
            make_dwell_report(
                dwell_start,
                dwell_stop,
                obsid,
                obs_events,
                outdir=Path(opt.web_out)
                / "events"
                / f"{year}"
                / f"dwell_{dwell_start}",
                redo=True,
            )
        make_summary_reports(
            bgd_events[ok], outdir=opt.web_out, data_root=opt.data_root
        )
        return

    # If the user has asked for a start time earlier than the end of the
    # table, delete possibly conflicting events in the table.
    if opt.start is not None:
        if start is not None:
            bgd_events = bgd_events[
                (bgd_events["dwell_datestart"] < CxoTime(opt.start).date)
                | (bgd_events["dwell_datestart"] > CxoTime(opt.stop).date)
            ]
        start = CxoTime(opt.start)
    if start is None:
        start = CxoTime(-7)

    new_events, last_proc_time = get_events(
        start, stop=opt.stop, outdir=opt.web_out, data_root=opt.data_root
    )

    if len(new_events) > 0:
        new_events = Table(new_events)

        # Filter to just the big events for emails and report page
        ok = significant_events(new_events)
        big_events = new_events[ok]

        for dwell_start in np.unique(big_events["dwell_datestart"]):
            obs_events = new_events[new_events["dwell_datestart"] == dwell_start]
            year = int(CxoTime(dwell_start).frac_year)
            event_outdir = (
                Path(opt.web_out) / "events" / f"{year}" / f"dwell_{dwell_start}"
            )
            obsid = obs_events["obsid"][0]
            url = f"{opt.web_url}/events/{year}/dwell_{dwell_start}"

            make_dwell_report(
                dwell_start,
                obs_events["dwell_datestop"][0],
                obsid,
                obs_events,
                outdir=event_outdir,
            )
            LOGGER.warning(f"HI BGD event in obsid {obsid} {url}")

            # Add another filter on the data for the emails to only include
            # events with more than 200 slot seconds.
            if np.any(obs_events["slot_seconds"] > 200) and len(opt.emails) > 0:
                send_mail(
                    LOGGER,
                    opt,
                    f"ACA HI BGD event in obsid {obsid}",
                    f"HI BGD in obsid {obsid} report at {url} with "
                    f"max duration {np.max(obs_events['duration']):.1f}"
                    f" and max slot seconds {np.max(obs_events['slot_seconds']):.1f}",
                    __file__,
                )

    if len(bgd_events) > 0:
        bgd_events = vstack([bgd_events, new_events])
    else:
        bgd_events = new_events

    if len(bgd_events) == 0:
        LOGGER.warning("No new or old events - Bailing out")
        return

    # Sort the table by date
    bgd_events.sort("datestart")

    # Add a null event at the end
    bgd_events.add_row()
    bgd_events[-1]["obsid"] = -1
    bgd_events[-1]["dwell_datestart"] = CxoTime(last_proc_time).date

    # Mark the significant events in the table as "has_report"
    ok = significant_events(bgd_events)
    bgd_events["has_report"] = False
    bgd_events["has_report"][ok] = True

    bgd_events.write(EVENT_ARCHIVE, format="ascii.ecsv", overwrite=True)

    # Make summary reports of the significant ones.
    make_summary_reports(bgd_events[ok], outdir=opt.web_out)


if __name__ == "__main__":
    main()
