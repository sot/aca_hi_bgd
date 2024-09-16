import argparse
import os
from pathlib import Path

import numpy as np

os.environ["MPLBACKEND"] = "Agg"

import json
import warnings

import agasc
import astropy.units as u
import matplotlib.pyplot as plt
import numba
from acdc.common import send_mail
from astropy.table import Table, vstack
from chandra_aca.transform import mag_to_count_rate
from chandra_aca.dark_subtract import get_aca_images
from cheta import fetch, fetch_sci
from cxotime import CxoTime
from jinja2 import Template
from kadi import events
from kadi.commands import get_starcats
from mica.archive import aca_l0
from mica.archive.aca_dark.dark_cal import get_dark_cal_props
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from ska_helpers.logging import basic_logger
from ska_helpers.run_info import log_run_info
from ska_matplotlib import plot_cxctime
from ska_numpy import interpolate

from aca_hi_bgd import __version__

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Warning: 'partition' will ignore the 'mask' of the MaskedColumn",
)


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


def exceeds_6x6(slot_data):
    ok = (
        (slot_data["QUALITY"] == 0)
        & (slot_data["IMGFUNC1"] == 1)
        & (slot_data["IMGSIZE"] == 6)
    )
    hits = np.zeros(len(slot_data), dtype=bool)
    hits = ok & (slot_data["BGDAVG"] > 200)
    return hits


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def get_events_6x6(slot_data):
    cand_crossings = []
    hits = exceeds_6x6(slot_data)
    overs = np.flatnonzero(hits)
    if len(overs) < 3:
        return []
    consec = consecutive(overs)
    for chunk in consec:
        if len(chunk) < 3:
            continue
        cross_time = slot_data["TIME"][chunk[0]]
        # Save candidate crossings
        cand_crossings.append(cross_time)

    return cand_crossings


def exceeds_8x8(slot_data):
    ok = (
        (slot_data["QUALITY"] == 0)
        & (slot_data["IMGFUNC1"] == 1)
        & (slot_data["IMGSIZE"] == 8)
    )
    hits = np.zeros(len(slot_data), dtype=bool)

    hits[ok] = slot_data["outer_min_7_magsub"][ok] > slot_data["threshold"][ok]
    return hits


def get_events_8x8(slot_data):
    cand_crossings = []
    hits = exceeds_8x8(slot_data)
    overs = np.flatnonzero(hits)
    if len(overs) == 0:
        return []
    consec = consecutive(overs)
    for chunk in consec:
        cross_time = slot_data["TIME"][chunk[0]]
        cand_crossings.append(cross_time)
    return cand_crossings


def get_candidate_crossings(slots_data):
    """
    Find times when BGDAVG crosses a threshold

    Check the BGDAVG in the image data in all slots for a dwell for values greater than
    threshold, and if the threshold crossings are consecutive for 3 or more
    samples, add to a list of candidate events.

    :param slots_data: dict of tables of L0 data including BGDAVG
    :returns: list of cxc times of potential events
    """
    # Check for background crossings
    cand_crossings = []
    for slot in range(8):
        slot_data = slots_data[slot]
        ok = (slot_data["QUALITY"] == 0) & (slot_data["IMGFUNC1"] == 1)

        # If this slot is mostly 6x6 data, use the old method of checking BGDAVG
        # against a temperate-scaled threshold
        if np.median(slot_data["IMGSIZE"][ok]) == 6:
            cand_crossings.extend(get_events_6x6(slot_data))

        # If this slot is mostly 8x8 data, use new new method of checking an
        # outer_min relative to the median outer_min for the dwell
        elif np.median(slot_data["IMGSIZE"][ok]) == 8:
            cand_crossings.extend(get_events_8x8(slot_data))

        else:
            # And if this slot is mostly 4x4 data, skip it
            continue

    return cand_crossings


def get_event_at_crossing(cross_time, slots_data):
    """
    Get background event data at each crossing

    Review BGDAVG around a high threshold crossing (from get_candidate_crossings)
    and count "slot seconds" in all slots in a time range around that event
    where those samples have values above a low threshold (100).

    The slot seconds are calculated in the range between 100 seconds before
    the `cross_time` to 300 seconds after the `cross_time`.

    :param cross_time: cxc time of high threshold crossing
    :param slots_data: dictionary of astropy tables of L0 data with BGDAVG
    :param thresh: low threshold in DN
    :returns: dict of an 'event' associated with the threshold crossings
    """
    sum_slot_seconds = 0
    event = {
        "tstart": cross_time,
        "tstop": cross_time,
        "max_bgd": 0,
        "slots_for_sum": {},
        "event_slots": {},
    }

    for slot in range(8):
        slot_data = slots_data[slot]
        if np.median(slot_data["IMGSIZE"]) == 6:
            hits = exceeds_6x6(slot_data)
        elif np.median(slot_data["IMGSIZE"]) == 8:
            hits = exceeds_8x8(slot_data)
        else:
            continue

        # To count as slot_seconds for the event, the record should be
        # valid data in the range of -100 to +300 over the threshold.

        count_ok = (
            (slot_data["TIME"] >= (cross_time - 100))
            & (slot_data["TIME"] <= (cross_time + 300))
            & (slot_data["QUALITY"] == 0)
            & (slot_data["IMGFUNC1"] == 1)
            & (hits)
        )
        if np.count_nonzero(count_ok) == 0:
            continue
        event["max_bgd"] = max(
            event["max_bgd"], np.max(slot_data["bgd"].data[count_ok])
        )
        imgsize = slot_data["IMGSIZE"].data[count_ok]
        dts = np.ones(len(imgsize)) * 1.025
        not4 = imgsize != 4
        dts[not4] = (imgsize[not4] - 4) * 1.025
        slot_seconds = np.sum((dts))
        if slot_seconds > 0:
            event["slots_for_sum"][slot] = 1
        sum_slot_seconds += slot_seconds

        consec = consecutive(np.flatnonzero(hits))
        for chunk in consec:
            if len(chunk) == 0:
                continue
            if (slot_data["TIME"][chunk[0]] <= cross_time) & (
                slot_data["TIME"][chunk[-1]] >= cross_time
            ):
                event["tstart"] = min([event["tstart"], slot_data["TIME"][chunk[0]]])
                # Set tstop to be the latest time in the chunk plus the image time.
                event["tstop"] = max(
                    [
                        event["tstop"],
                        slot_data["TIME"][chunk[-1]]
                        + (slot_data["IMGSIZE"][chunk[-1]] - 4) * 1.025,
                    ]
                )
                event["event_slots"][slot] = 1
                # If we had a match in this chunk, break out of the loop over chunks
                break
    event["slot_seconds"] = sum_slot_seconds
    return event


def combine_events(events, tol=30):
    """
    Combine events

    For a list of dictionaries of events, combine them if they are continuous
    with gaps up to `tol`.  Dictionaries are combined by extending the end time
    of a "previous" event (and associated overall duration).  To avoid double-
    counting, only the the maximum value of the slot_seconds metric is assocated
    with the new comabined interval of the event.

    :param events: list of dictionaries of events
    :param tol: tolerance in seconds.  raw events separated by more than this will
                not be combined
    :returns: astropy table of combined events
    """

    # Not sure how to set width of a string column in astropy table
    # so go through some silly machinations here to get the string
    # columns always at least 15 chars.
    events.append(events[0].copy())
    events[-1]["slots"] = " " * 15
    events[-1]["slots_for_sum"] = " " * 15
    events = Table(events)
    events = events[0:-1]

    events.sort("event_datestart")
    combined = Table(events[0])
    for e in events[1:]:
        last_event = combined[-1]
        if (e["dwell_datestart"] == last_event["dwell_datestart"]) and (
            (e["event_tstart"] - tol) <= last_event["event_tstop"]
        ):
            last_event["event_tstop"] = e["event_tstop"]
            last_event["duration"] = (
                last_event["event_tstop"] - last_event["event_tstart"]
            )
            # update to include any new slots
            slots = list(set(last_event["slots"].split(",") + e["slots"].split(",")))
            last_event["slots"] = ",".join(sorted([str(s) for s in slots]))
            if e["slot_seconds"] > last_event["slot_seconds"]:
                slots_for_sum = list(
                    set(
                        last_event["slots_for_sum"].split(",")
                        + e["slots_for_sum"].split(",")
                    )
                )
                last_event["slots_for_sum"] = ",".join(
                    sorted([str(s) for s in slots_for_sum])
                )
                for col in ["cross_time", "slot_seconds"]:
                    last_event[col] = e[col]
        else:
            combined = vstack([combined, Table(e)])
    return combined


def get_events(start, stop=None, outdir=None):
    """
    Get high background events in a time range

    Loop over the kadi dwells in a time range and check for high background
    events (using L0 BGDAVG) in each dwell.

    :param start: start time for earliest dwell to check
    :param stop: stop time used in filter for end of dwell range to check
    :returns: tuple (astropy table of high background events, end time of checked range)
    """

    start = CxoTime(start)
    stop = CxoTime(stop)
    dwells = events.dwells.filter(start__gt=start.date, stop=stop)
    bgd_events = []
    if len(dwells) == 0:
        return bgd_events, start.date

    stop_with_data = start.date

    if (Path(outdir) / "dwell_metrics.csv").exists():
        dwell_metrics_tab = Table.read(Path(outdir) / "dwell_metrics.csv")
        # Convert the table to a list of dictionaries
        names = dwell_metrics_tab.colnames
        dwell_metrics = [dict(zip(names, row)) for row in dwell_metrics_tab]
    else:
        dwell_metrics = []

    for d in dwells:

        if len(dwell_metrics) > 0:
            dwell_starts = [d["dwell"] for d in dwell_metrics]
            if d.start in dwell_starts:
                row_dict = dwell_metrics[dwell_starts.index(d.start)]
                hit = False
                for slot in range(8):
                    if row_dict[f'max_s{slot}'] > row_dict[f'threshold_s{slot}']:
                        hit = True
                if hit is False:
                    LOGGER.info(f"Skipping dwell {d} obsid {d.get_obsid()} no hits")
                    continue

        if d.get_obsid() == 0:
            LOGGER.info(f"Skipping dwell {d} obsid {d.get_obsid()} in anomaly or recovery.")
            continue

        try:
            dwell_events, stop, slot_metrics = get_dwell_events(d)
        except IndexError:
            LOGGER.info(f"Skipping dwell {d} obsid {d.get_obsid()} no data")
            continue

        if stop is None:
            if (CxoTime.now() - CxoTime(d.stop)) < 7 * u.day:
                break
            else:
                continue
        else:
            stop_with_data = stop

        # unwrap the slot_metrics and save them
        dwell_metric = {'dwell': d.start, 'obsid': d.get_obsid()}
        for slot in slot_metrics:
            for col in slot_metrics[slot]:
                dwell_metric[f'{col}_s{slot}'] = slot_metrics[slot][col]
        dwell_metrics.append(dwell_metric)
        if len(dwell_metrics) > 1:
            # make the outdir
            Path(outdir).mkdir(parents=True, exist_ok=True)
            Table(dwell_metrics).write(Path(outdir) / 'dwell_metrics.csv', overwrite=True)

        if len(dwell_events) > 0:
            dwell_events = combine_events(dwell_events)
            event_outdir = Path(outdir) / "events" / f"dwell_{d.start}"
            make_event_report(d.start, d.get_obsid(), dwell_events, outdir=event_outdir)
            json.dump(
                dwell_events.as_array().tolist(),
                open((Path(event_outdir) / "events.json"), "w"),
            )
            if len(bgd_events) > 0:
                bgd_events = vstack([Table(bgd_events), dwell_events])
            else:
                bgd_events = dwell_events

    return bgd_events, stop_with_data


def get_outer_min(imgs, rank=0):
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

    outer_min = np.sort(imgs.reshape(img_len, 64)[tile_mask].reshape(img_len, used_pix), axis=-1)[:, rank]

    return outer_min


def get_max_of_mins(slots_data, col):
    # Get some debug data to try to calibrate what should be used for a
    # threshold for the BGDAVG and outer_min values.
    max_of_mins = 0
    for slot in range(8):
        if slot not in slots_data:
            continue
        slot_data = slots_data[slot]
        ok = (slot_data["QUALITY"] == 0) & (slot_data["IMGFUNC1"] == 1)

        # calculate rolling minumum with a 3 sample window
        if len(slot_data[col][ok]) < 3:
            continue
        rolling_min = np.min(sliding_window_view(slot_data[col][ok], 2), axis=-1)
        max_of_mins = max(max_of_mins, np.max(rolling_min))
    return max_of_mins


def get_star_contrib(mag):
    counts = mag_to_count_rate(mag)
    # mag_p = [2.5e-10, 3.0e-5, 0]
    mag_p = [1.55e-4, 0]
    mag_error_p = [1.822e-05, 2.393]
    mag_contrib = np.clip(np.polyval(mag_p, counts), 0, None)
    mag_contrib_err = np.clip(np.polyval(mag_error_p, counts), 0, None)
    return mag_contrib, mag_contrib_err


def get_background_data(slot_data, mag):
    ok_8 = (
        (slot_data["IMGSIZE"] == 8)
        & (slot_data["QUALITY"] == 0)
        & (slot_data["IMGFUNC1"] == 1)
    )

    bgds = {}

    imgs_8x8_bgsub = slot_data['IMGBGSUB']

    outer_min_7 = np.zeros(len(slot_data))
    outer_min_7[ok_8] = get_outer_min(imgs_8x8_bgsub[ok_8], rank=7)
    if np.count_nonzero(ok_8) > 2:
        outer_min_7[~ok_8] = interpolate(
            outer_min_7[ok_8], slot_data["TIME"][ok_8], slot_data["TIME"][~ok_8]
        )
    bgds["outer_min_7"] = outer_min_7

    # Subtract off an estimate of the star or fid contribution by magnitude
    mag_contrib, mag_contrib_err = get_star_contrib(mag)
    bgds["outer_min_7_magsub"] = outer_min_7 - mag_contrib

    # Just capture a threshold here.
    bgds["threshold"] = np.ones_like(outer_min_7) * 47
    bgds["threshold"][ok_8] = np.max([47, 6 * mag_contrib_err])

    bgds["bgd"] = np.where(
        slot_data["IMGSIZE"] == 8, bgds["outer_min_7_magsub"], slot_data["BGDAVG"]
    )
    return bgds


def get_slot_mags(time):
    """
    Get the slot magnitudes for a time

    :param time: time to get slot magnitudes
    :returns: dict of slot magnitudes
    """
    # Get catalog magnitudes for the things in this dwell
    starcats = get_starcats(CxoTime(time).secs - (3600 * 4), CxoTime(time).secs)
    if len(starcats) == 0:
        LOGGER.info(f"No starcat data for {CxoTime(time).date}")
        return {}
    starcat = starcats[-1]
    guide_cat = starcat[starcat["type"] != "ACQ"]

    slot_mag = {}
    for slot in range(8):
        guide_slots = guide_cat[guide_cat["slot"] == slot]
        if len(guide_slots) != 1:
            slot_mag[slot] = 15
        else:
            guide_slot = guide_slots[0]
            if np.in1d(guide_slot["type"], ["BOT", "GUI"]):
                star = agasc.get_star(
                    guide_slot["id"], agasc_file="/proj/sot/ska/data/agasc/agasc1p8.h5"
                )
                slot_mag[slot] = star["MAG_ACA"]
            else:
                slot_mag[slot] = guide_slot["mag"]
    return slot_mag


def get_slots_metrics(slots_data):

    slots_metrics = {}
    for slot in range(8):
        if slot not in slots_data:
            continue
        slot_data = slots_data[slot]
        ok = (slot_data["QUALITY"] == 0) & (slot_data["IMGFUNC1"] == 1)
        if np.median(slot_data["IMGSIZE"][ok]) == 6:
            ok6 = ((slot_data["QUALITY"] == 0)
                     & (slot_data["IMGFUNC1"] == 1)
                     & (slot_data["IMGSIZE"] == 6))
            slots_metrics[slot] = {
                "bg_col": "BGDAVG",
                "threshold": 200,
                "max": slot_data["BGDAVG"][ok6].max()
            }
        elif np.median(slot_data["IMGSIZE"][ok]) == 8:
            ok8 = ((slot_data["QUALITY"] == 0)
                   & (slot_data["IMGFUNC1"] == 1)
                   & (slot_data["IMGSIZE"] == 8))
            slots_metrics[slot] = {
                "bg_col": "outer_min_7_magsub",
                "max": slot_data["outer_min_7_magsub"][ok8].max(),
                "threshold": np.median(slot_data['threshold'][ok8]),
            }
        else:
            continue
    return slots_metrics



def get_dwell_events(dwell):
    """
    Review a single dwell for high background events

    :param dwell: kadi dwell event
    :returns: tuple (list of dictionaries of high background events,
                     dwell stop of last checked dwell)
    """

    d = dwell
    try:
        obsid = d.get_obsid()
    except Exception:
        obsid = 0

    LOGGER.info(f"Checking dwell {d} obsid {obsid} for events")

    slot_mag = get_slot_mags(dwell.tstart)

    bgd_events = []

    slots_data = get_aca_images(d.start, d.stop, source='mica')
    for slot in slots_data:
        mag = slot_mag[slot] if slot in slot_mag else 15
        bgds = get_background_data(slots_data[slot], mag)
        for key in bgds:
            slots_data[slot][key] = bgds[key]

    slots_metrics = get_slots_metrics(slots_data)

    # Check that the image data is complete for the dwell.
    # This assumes that it is sufficient to check slot 3
    if (len(slots_data[3]) == 0) or (
        CxoTime(d.stop).secs - slots_data[3]["TIME"][-1]
    ) > 60:
        LOGGER.info(f"Stopping review of dwells at dwell {d.start}, missing image data")
        return [], None, {}

    # Get Candidate crossings
    cand_crossings = get_candidate_crossings(slots_data)

    # If there are candidate crossings, get the pitch of this dwell
    pitch = -999
    if len(cand_crossings) > 0:
        pitchs = fetch.Msid("DP_PITCH", d.start, d.stop, stat="5min")
        pitch = np.median(pitchs.vals)


    # Review the crossings and check for slot seconds
    for cross_time in np.unique(cand_crossings):
        event = get_event_at_crossing(cross_time, slots_data)

        if len(event["event_slots"]) == 0:
            continue

        e = {
            "slots": ",".join(sorted([str(s) for s in event["event_slots"]])),
            "slots_for_sum": ",".join(sorted([str(s) for s in event["slots_for_sum"]])),
            "n_slots": len(event["event_slots"].keys()),
            "obsid": obsid,
            "slot_seconds": event["slot_seconds"],
            "cross_time": cross_time,
            "dwell_tstart": d.tstart,
            "dwell_datestart": d.start,
            "max_bgd_dn": event["max_bgd"],
            "duration": event["tstop"] - event["tstart"],
            "event_tstart": event["tstart"],
            "event_tstop": event["tstop"],
            "event_datestart": CxoTime(event["tstart"]).date,
            "pitch": pitch,
        }
        LOGGER.info(
            f"Updating with {e['duration']:.1f}s raw event in {obsid} at {e['event_datestart']}"
        )
        bgd_events.append(e)
    return bgd_events, d.stop, slots_metrics


def make_event_report(dwell_start, obsid, obs_events, outdir=".", redo=False):
    # Do the per-obsid plot and report making

    if not Path(outdir).exists():
        # Make the directory if it doesn't exist using Path
        Path(outdir).mkdir(parents=True, exist_ok=True)
    elif redo is False:
        return

    events = []
    for e in obs_events:
        event = dict(zip(e.colnames, e.as_void(), strict=False))
        event["bgdplot"] = plot_bgd(event, outdir)
        event["aokalstr"] = plot_aokalstr(event, outdir)
        event["imgrows"] = make_images(
            event["cross_time"] - 25, event["cross_time"] + 100, outdir
        )
        events.append(event)

    LOGGER.info(f"Making report for {outdir}")
    file_dir = Path(__file__).parent
    obs_template = Template(open(file_dir / "per_obs_template.html", "r").read())
    page = obs_template.render(dwell_start=dwell_start, obsid=obsid, events=events)
    f = open(Path(outdir) / "index.html", "w")
    f.write(page)
    f.close()


def plot_bgd(e, edir):
    """
    Make a plot of background over an event and save to `edir`.

    Presently plots over range from 25 seconds before high threshold crossing to
    100 seconds after high threshold crossing.

    :param e: dictionary with times of background event
    :param edir: directory for plots
    """

    fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))

    slot_mag = get_slot_mags(e["dwell_tstart"])

    start = e["cross_time"] - 25
    stop = e["cross_time"] + 100
    slots_data = get_aca_images(start, stop,
                                source='mica')
    for slot in range(8):

        slot_data = slots_data[slot]

        if len(slot_data["TIME"]) == 0:
            raise ValueError
        mag = slot_mag[slot] if slot in slot_mag else 15
        bgds = get_background_data(slot_data, mag)
        for key in bgds:
            slot_data[key] = bgds[key]
        ok = (slot_data["QUALITY"] == 0) & (slot_data["IMGFUNC1"] == 1)
        if np.count_nonzero(ok) > 0:
            plot_cxctime(
                slot_data["TIME"][ok],
                slot_data["BGDAVG"][ok],
                ".",
                markersize=5,
                label=f"slot {slot}",
                ax=ax[0],
            )
            ax[0].tick_params(axis="both", which="major", labelsize=7, pad=0)
            ax[0].set_ylabel("BGDAVG (DN)", fontsize="x-small")
            ax[0].grid(True)
            plot_cxctime(
                slot_data["TIME"][ok],
                slot_data["outer_min_7_magsub"][ok] * 1.696 / 5.0,
                ".",
                markersize=5,
                label=f"slot {slot}",
                ax=ax[1],
            )
            ax[1].set_ylabel("mag & dark sub 8th outer min (e-/s)", fontsize="x-small")
            ax[1].grid(True)
            ax[1].tick_params(axis="both", which="major", labelsize=7, pad=0)
    ax[0].set_title(
        "BGDAVG obsid {}\n start {}".format(
            e["obsid"], CxoTime(e["event_tstart"]).date
        ),
        fontsize="x-small",
    )
    ax[1].set_title(
        f"corr background 8th outer min {e['obsid']}\n start {CxoTime(e['event_tstart']).date}",
        fontsize="x-small",
    )

    handles, labels = ax[0].get_legend_handles_labels()
    labels, handles = zip(
        *sorted(zip(labels, handles, strict=False), key=lambda t: t[0]), strict=False
    )
    fig.legend(
        handles,
        labels,
        numpoints=1,
        fontsize="x-small",
        loc="lower right",
        bbox_to_anchor=(1, -0.022),
        bbox_transform=fig.transFigure,
        ncol=len(labels),
    )
    plt.tight_layout()
    plt.margins(0.05)
    ax[0].set_ylim([0, 1100])
    ylims1 = ax[1].get_ylim()
    ax[1].set_ylim([0, np.max([250, ylims1[1]])])
    filename = "bgdavg_{}.png".format(e["event_datestart"])
    plt.savefig(Path(edir) / filename, dpi=150)
    plt.close()
    return filename


def plot_aokalstr(e, edir):
    """
    Plot AOKALSTR over the event with times in the supplied dictionary.

    :param e: dictionary with times of background event
    :param edir: directory for plots
    """
    plt.figure(figsize=(3, 2.5))
    aokalstr = fetch.Msid("AOKALSTR", e["event_tstart"] - 100, e["event_tstop"] + 100)
    aokalstr.plot()
    plt.ylim([0, 9])
    plt.grid()
    plt.title("AOKALSTR", fontsize="x-small")
    filename = "aokalstr_{}.png".format(e["event_datestart"])
    plt.tight_layout()
    plt.tick_params(axis="both", which="major", labelsize=8)
    plt.savefig(Path(edir) / filename, dpi=150)
    plt.close()
    return filename


def make_images(start, stop, outdir="out", max_images=200):
    """
    Make pixel plots of the a time range and write out to a directory.

    :param start: start of time range (Chandra.Time compatible)
    :param stop: end of time range (Chandra.Time compatible)
    :param outdir: output directory
    :param max_images: stop making image files if more more than this number
                       have been made
    """
    start = CxoTime(start)
    stop = CxoTime(stop)

    slot_mag = get_slot_mags(start.secs)

    slots_data = get_aca_images(start.secs - 10,
                                stop.secs + 10, source='mica')
    for slot in range(8):
        mag = slot_mag[slot] if slot in slot_mag else 15
        slots_data[slot] = slots_data[slot]
        slots_data[slot]["SLOT"] = slot
        bgds = get_background_data(slots_data[slot], mag)
        for key in bgds:
            slots_data[slot][key] = bgds[key]

    # Get a list of all the unique times in the set
    times = np.unique(
        np.concatenate([slots_data[slot]["TIME"].data for slot in range(8)])
    )
    times = times[(times >= (start.secs - 4.5)) & (times <= (stop.secs + 4.5))]

    SIZE = 96
    rows = []
    for time in times:
        row = {"rowsecs": time, "rowdate": CxoTime(time).date, "slots": []}
        slot_imgs = []
        for slot in range(8):
            # Find last complete row at or before this time
            last_idx = np.flatnonzero(slots_data[slot]["TIME"] <= time)[-1]
            dat = slots_data[slot][last_idx]
            imgraw = dat["IMG"].filled(0).reshape(8, 8)
            #sz = dat["IMGSIZE"]
            sz = 8
            #if sz < 8:
            #    imgraw = imgraw[:sz, :sz]
            imgraw = np.clip(imgraw, a_min=10, a_max=None)
            if np.all(imgraw == 10) or dat["IMGFUNC1"] != 1:
                pixvals = np.zeros((sz, sz))
                np.fill_diagonal(pixvals, 255)
            else:
                # Log scale the images but use a fixed low end at 4
                logimg = np.log(imgraw)
                pixvals = 255 * (logimg - 4) / (np.max(logimg) - 4)

            # Scale the image because the browser isn't great at it
            pixvals = np.kron(pixvals, np.ones((int(SIZE / sz), int(SIZE / sz))))
            # Set a border for stale data
            img = (
                255 * np.ones((108, 108))
                if dat["TIME"] < row["rowsecs"]
                else np.zeros((108, 108))
            )

            img[6:102, 6:102] = pixvals
            slot_imgs.append(img)
            # bgcolor = 'gray' if dat['TIME'] < row['rowsecs'] else 'white'
            row["slots"].append(
                {
                    "stale": (dat["TIME"] < row["rowsecs"]),
                    "slot": slot,
                    "time": dat["TIME"],
                    "rowsecs": row["rowsecs"],
                    "bgd": dat["bgd"],
                    "bgdavg": dat["BGDAVG"],
                    "outer_min_7_magsub": np.clip(dat["outer_min_7_magsub"], 0, None),
                    "imgfunc1": dat["IMGFUNC1"],
                }
            )

        im = Image.fromarray(np.hstack(slot_imgs)).convert("RGB")
        imgfilename = "piximg_{}.png".format(row["rowsecs"])
        im.save(Path(outdir) / imgfilename, "PNG")
        row["img"] = imgfilename
        rows.append(row)
        if len(rows) > max_images:
            break
    return rows


def make_summary_reports(bgd_events, outdir="."):
    """
    Make event reports

    For a table of background events, make a high level summary report
    by obsid and make an individual obsid report for each.

    :param bgd_events: astropy table with times and obsids of events
    :param outdir: output directory
    :param redo: if set, remake plots and reports if they already exist
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    bgd_events = bgd_events[bgd_events["obsid"] != 0]
    bgd_events = bgd_events[bgd_events["obsid"] != -1]

    dwell_events = []
    for dwell_start in np.unique(bgd_events["dwell_datestart"]):
        events = bgd_events[bgd_events["dwell_datestart"] == dwell_start]
        events["index"] = np.arange(len(events))

        # ignore the obsid == 0 case(s) for now
        if np.all(events["obsid"] == 0):
            continue

        slots = {}
        for e in events:
            for s in e["slots"].split(","):
                slots[int(s)] = 1

        # Save the events, the intervals, and some other useful stuff for the per-dwell table
        obs = {
            "events": events.copy(),
            "dwell_datestart": dwell_start,
            "datestart": CxoTime(events[0]["event_tstart"]).date,
            "n_events": len(events),
            "max_dur": np.max(events["duration"]),
            "max_slot_secs": np.max(events["slot_seconds"]),
            "n_slots": len(slots),
            "obsid": events["obsid"][0],
            "reldir": f"events/dwell_{dwell_start}",
            "dir": Path(outdir) / "events" / f"dwell_{dwell_start}",
            "pitch": events[0]["pitch"],
        }
        dwell_events.append(obs)

    dwell_events = sorted(dwell_events, key=lambda i: i["datestart"])
    dwell_events = dwell_events[::-1]

    file_dir = Path(__file__).parent
    template = Template(open(file_dir / "top_level_template.html", "r").read())
    page = template.render(obs_events=dwell_events)
    f = open(Path(outdir) / "index.html", "w")
    f.write(page)
    f.close()


def review_and_send_email(events, opt):
    # For the new events, first filter down to ones worth notifying about.
    # Let's say that's at least 3 slots or at least 20 seconds.
    # And only warn on legit obsids
    ok = ( ~np.in1d(events["obsid"], [0, -1])
            & (events["n_slots"] >= 3)
            | (events["duration"] >= 20)
    )
    events = events[ok]

    for dwell_start in np.unique(events["dwell_datestart"]):
        obs_events = events[events["dwell_datestart"] == dwell_start]
        obsid = obs_events["obsid"][0]
        url = f"{opt.web_url}/events/dwell_{dwell_start}"
        send_mail(
            LOGGER,
            opt,
            f"ACA HI BGD event in obsid {obsid}",
            f"HI BGD in obsid {obsid} report at {url}",
            __file__,
        )


def main(args=None):
    """
    Do high background processing

    Review kadi dwells for new high background events, update a text file table of
    those events, make reports, and notify via email as needed.
    """

    opt = get_opt().parse_args(args)
    log_run_info(LOGGER.info, opt, version=__version__)

    EVENT_ARCHIVE = Path(opt.data_root) / "bgd_events.dat"
    Path(opt.data_root).mkdir(parents=True, exist_ok=True)
    start = None

    bgd_events = []
    if Path(EVENT_ARCHIVE).exists():
        bgd_events = Table.read(EVENT_ARCHIVE, format="ascii")
    if len(bgd_events) > 0:
        start = CxoTime(bgd_events["dwell_datestart"][-1])
        # Remove any bogus events from the real list
        bgd_events = bgd_events[bgd_events["obsid"] != -1]
        bgd_events["slots"] = bgd_events["slots"].astype(str)
        bgd_events["slots_for_sum"] = bgd_events["slots_for_sum"].astype(str)

    # If the user has asked for a start time earlier than the end of the
    # table, delete any rows after the supplied start time
    if opt.start is not None:
        if start is not None:
            if CxoTime(opt.start).secs < start.secs:
                bgd_events = bgd_events[
                    bgd_events["dwell_datestart"] < CxoTime(opt.start).date
                ]
        start = CxoTime(opt.start)
    if start is None:
        start = CxoTime(-7)

    new_events, stop = get_events(start, stop=opt.stop, outdir=opt.web_out)
    if len(new_events) > 0:
        new_events = Table(new_events)
        if len(opt.emails) > 0:
            review_and_send_email(events=new_events, opt=opt)

        ok = ~np.in1d(new_events["obsid"], [0, -1])
        for dwell_start in np.unique(new_events["dwell_datestart"][ok]):
            obs_events = new_events[new_events["dwell_datestart"] == dwell_start]
            obsid = obs_events["obsid"][0]
            url = f"{opt.web_url}/events/dwell_{dwell_start}"
            LOGGER.warning(f"HI BGD event at in obsid {obsid} {url}")

    if len(bgd_events) > 0:
        bgd_events = vstack([bgd_events, new_events])
    else:
        bgd_events = new_events

    if len(bgd_events) == 0:
        LOGGER.warning("No new or old events - Bailing out")
        return

    # Add a null event at the end
    bgd_events.add_row()
    bgd_events[-1]["obsid"] = -1
    bgd_events[-1]["dwell_datestart"] = CxoTime(stop).date

    bgd_events.write(EVENT_ARCHIVE, format="ascii", overwrite=True)

    make_summary_reports(bgd_events, outdir=opt.web_out)


if __name__ == "__main__":
    main()
