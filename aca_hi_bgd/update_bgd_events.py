import os
from pathlib import Path
import argparse
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from astropy.table import Table, vstack
from jinja2 import Template

import pyyaks.logger
from Ska.Numpy import interpolate
from Chandra.Time import DateTime
from mica.archive import aca_l0
from kadi import events
from Ska.Matplotlib import plot_cxctime
from Ska.engarchive import fetch
from acdc.common import send_mail


logger = None


def get_opt(args=None):
    parser = argparse.ArgumentParser(description="High Background event finder")
    parser.add_argument("--start", help="Start date")
    parser.add_argument(
        "--data-root",
        default="/proj/sot/ska/data/aca_hi_bgd_mon",
        help="Output data directory",
    )
    parser.add_argument("--web-out", default="/proj/sot/ska/www/ASPECT/aca_hi_bgd_mon/")
    parser.add_argument(
        "--web-url", default="https://cxc.harvard.edu/mta/ASPECT/aca_hi_bgd_mon"
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=pyyaks.logger.INFO,
        help="Logging level (default=info)",
    )
    parser.add_argument(
        "--email",
        action="append",
        dest="emails",
        default=[],
        help="Email address for notificaion",
    )
    args = parser.parse_args()
    return args


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def get_slot_image_data(start, stop, slot):
    slot_data = aca_l0.get_slot_data(
        start,
        stop,
        imgsize=[4, 6, 8],
        slot=slot,
        columns=["TIME", "BGDAVG", "IMGFUNC1", "QUALITY", "IMGSIZE"],
    )
    return Table(slot_data)


def get_candidate_crossings(slots_data, threshold=200):
    """
    Check the BGDAVG in the image data in all slots for a dwell for values greater than
    threshold, and if the threshold crossings are consecutive for 3 or more
    samples, add to a list of candidate events.

    :param slots_data: dict of tables of L0 data including BGDAVG
    :param threshold: threshold in DN
    :returns: list of cxc times of potential events
    """
    # Check for background crossings
    cand_crossings = []
    for slot in range(8):
        slot_data = slots_data[slot]
        ok = (slot_data["QUALITY"] == 0) & (slot_data["IMGFUNC1"] == 1)
        overs = np.flatnonzero(ok & (slot_data["BGDAVG"] > threshold))
        if len(overs) < 3:
            continue
        consec = consecutive(overs)
        for chunk in consec:
            if len(chunk) < 3:
                continue
            cross_time = slot_data["TIME"][chunk[0]]
            # Save candidate crossings
            cand_crossings.append(cross_time)
    return cand_crossings


def get_event_at_crossing(cross_time, slots_data, thresh=100):
    """
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

        # To count as slot_seconds for the event, the record should be
        # valid data in the range of -100 to +300 over the threshold.
        count_ok = (
            (slot_data["TIME"] >= (cross_time - 100))
            & (slot_data["TIME"] <= (cross_time + 300))
            & (slot_data["QUALITY"] == 0)
            & (slot_data["IMGFUNC1"] == 1)
            & (slot_data["BGDAVG"] > thresh)
        )
        if np.count_nonzero(count_ok) == 0:
            continue
        event["max_bgd"] = max(
            event["max_bgd"], np.max(slot_data["BGDAVG"].data[count_ok])
        )
        imgsize = slot_data["IMGSIZE"].data[count_ok]
        dts = np.ones(len(imgsize)) * 1.025
        not4 = imgsize != 4
        dts[not4] = (imgsize[not4] - 4) * 1.025
        slot_seconds = np.sum((dts))
        if slot_seconds > 0:
            event["slots_for_sum"][slot] = 1
        sum_slot_seconds += slot_seconds

        # Interpolate over bad/missing data again because we didn't save this
        # from the last go
        bgdavg = slot_data["BGDAVG"].data.copy()
        ok = (slot_data["QUALITY"] == 0) & (slot_data["IMGFUNC1"] == 1)
        bgdavg = interpolate(bgdavg[ok], slot_data["TIME"][ok], slot_data["TIME"])
        # Clip
        bgdavg.clip(0, 1023)

        consec = consecutive(np.flatnonzero(bgdavg >= thresh))
        for chunk in consec:
            if len(chunk) == 0:
                continue
            if (slot_data["TIME"][chunk[0]] <= cross_time) & (
                slot_data["TIME"][chunk[-1]] >= cross_time
            ):
                event["tstart"] = min([event["tstart"], slot_data["TIME"][chunk[0]]])
                event["tstop"] = max([event["tstop"], slot_data["TIME"][chunk[-1]]])
                event["event_slots"][slot] = 1
                # If we had a match in this chunk, break out of the loop over chunks
                break
    event["slot_seconds"] = sum_slot_seconds
    return event


def combine_events(events, tol=30):
    """
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
    events = Table(events)
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
            if e["slot_seconds"] > last_event["slot_seconds"]:
                for col in ["cross_time", "slot_seconds", "slots_for_sum"]:
                    last_event[col] = e[col]
        else:
            combined = vstack([combined, Table(e)])
    return combined


def get_events(start, stop=None):
    """
    Loop over the kadi dwells in a time range and check for high background
    events (using L0 BGDAVG) in each dwell.

    :param start: start time for earliest dwell to check
    :param stop: stop time used in filter for end of dwell range to check
    :returns: tuple (astropy table of high background events, end time of checked range)
    """
    global logger
    if logger is None:
        logger = pyyaks.logger.get_logger(level=pyyaks.logger.DEBUG)
    start = DateTime(start)
    stop = DateTime(stop)
    dwells = events.dwells.filter(start__gt=start.date, stop=stop)
    bgd_events = []
    if len(dwells) == 0:
        return bgd_events, start.date

    stop_with_data = start.date
    for d in dwells:
        dwell_events, stop = get_dwell_events(d)
        if stop is None:
            if (DateTime() - DateTime(d.stop)) < 7:
                break
            else:
                continue
        else:
            stop_with_data = stop
        if len(dwell_events) > 0:
            dwell_events = combine_events(dwell_events)
            if len(bgd_events) > 0:
                bgd_events = vstack([Table(bgd_events), dwell_events])
            else:
                bgd_events = dwell_events

    return bgd_events, stop_with_data


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

    logger.info(f"Checking dwell {d} obsid {obsid} for events")

    bgd_events = []

    # First, get all the image data
    slots_data = {}
    for slot in range(8):
        slots_data[slot] = get_slot_image_data(d.start, d.stop, slot)

    # Check that the image data is complete for the dwell.
    # This assumes that it is sufficient to check slot 3
    if (len(slots_data[3]) == 0) or (
        DateTime(d.stop).secs - slots_data[3]["TIME"][-1]
    ) > 60:
        logger.info(f"Stopping review of dwells at dwell {d.start}, missing image data")
        return [], None

    # Get Candidate crossings
    cand_crossings = get_candidate_crossings(slots_data)

    # If there are candidate crossings, get the pitch of this dwell
    pitch = -999
    if len(cand_crossings) > 0:
        pitchs = fetch.Msid("DP_PITCH", d.start, d.stop, stat="5min")
        pitch = np.median(pitchs.vals)

    # Review the crossings and check for slot seconds
    for cross_time in cand_crossings:
        event = get_event_at_crossing(cross_time, slots_data)

        if event["slot_seconds"] >= 20:
            if len(event["event_slots"]) == 0:
                raise ValueError

            e = {
                "slots": ",".join([str(s) for s in event["event_slots"]]),
                "slots_for_sum": ",".join([str(s) for s in event["slots_for_sum"]]),
                "obsid": obsid,
                "slot_seconds": event["slot_seconds"],
                "cross_time": cross_time,
                "dwell_tstart": d.tstart,
                "dwell_datestart": d.start,
                "max_bgd": event["max_bgd"],
                "duration": event["tstop"] - event["tstart"],
                "event_tstart": event["tstart"],
                "event_tstop": event["tstop"],
                "event_datestart": DateTime(event["tstart"]).date,
                "pitch": pitch,
            }
            logger.info(
                f"Updating with {e['duration']} raw event in {obsid} at {e['event_datestart']}"
            )
            bgd_events.append(e)
    return bgd_events, d.stop


def plot_bgd(e, edir):
    """
    Make a plot of BGDAVG over an event and save to `edir`.

    Presently plots over range from 100 seconds before high threshold crossing to
    300 seconds after high threshold crossing.

    :param e: dictionary with times of background event
    :param edir: directory for plots
    """
    slots = [int(s) for s in e["slots_for_sum"].split(",")]
    plt.figure(figsize=(4, 4.5))
    max_bgd = 0
    for slot in slots:
        l0_telem = aca_l0.get_slot_data(
            e["cross_time"] - 100, e["cross_time"] + 300, imgsize=[4, 6, 8], slot=slot
        )
        if len(l0_telem["TIME"]) == 0:
            raise ValueError
        ok = (l0_telem["QUALITY"] == 0) & (l0_telem["IMGFUNC1"] == 1)
        if np.count_nonzero(ok) > 0:
            plot_cxctime(
                l0_telem["TIME"][ok], l0_telem["BGDAVG"][ok], ".", label=f"slot {slot}"
            )
            max_bgd = max([max_bgd, np.max(l0_telem["BGDAVG"][ok])])
    plt.title(
        "Hi BGD obsid {}\n start {}".format(
            e["obsid"], DateTime(e["event_tstart"]).date
        ),
        fontsize="small",
    )
    plt.legend(numpoints=1, fontsize="x-small")
    plt.tight_layout()
    plt.margins(0.05)
    plt.ylim([-20, 1100])
    plt.grid()
    filename = "bgdavg_{}.png".format(e["event_datestart"])
    plt.savefig(os.path.join(edir, filename))
    plt.close()
    return filename, max_bgd


def plot_aokalstr(e, edir):
    """
    Plot AOKALSTR over the event with times in the supplied dictionary.

    :param e: dictionary with times of background event
    :param edir: directory for plots
    """
    plt.figure(figsize=(4, 3))
    aokalstr = fetch.Msid("AOKALSTR", e["event_tstart"] - 100, e["event_tstop"] + 100)
    aokalstr.plot()
    plt.ylim([0, 9])
    plt.grid()
    plt.title("AOKALSTR")
    filename = "aokalstr_{}.png".format(e["event_datestart"])
    plt.savefig(os.path.join(edir, filename))
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
    start = DateTime(start)
    stop = DateTime(stop)
    slotdata = {}
    for slot in range(8):
        slotdata[slot] = Table(
            aca_l0.get_slot_data(start.secs - 20, stop.secs + 20, slot=slot).data
        )
        slotdata[slot]["SLOT"] = slot

    # Get a list of all the unique times in the set
    times = np.unique(
        np.concatenate([slotdata[slot]["TIME"].data for slot in range(8)])
    )
    times = times[(times >= (start.secs - 4.5)) & (times <= (stop.secs + 4.5))]

    SIZE = 96
    rows = []
    for time in times:
        row = {"rowsecs": time, "rowdate": DateTime(time).date, "slots": []}
        slot_imgs = []
        for slot in range(8):
            # Find last complete row at or before this time
            last_idx = np.flatnonzero(slotdata[slot]["TIME"] <= time)[-1]
            dat = slotdata[slot][last_idx]
            imgraw = dat["IMGRAW"].reshape(8, 8)
            sz = dat["IMGSIZE"]
            if sz < 8:
                imgraw = imgraw[:sz, :sz]
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
                    "bgdavg": dat["BGDAVG"],
                    "imgfunc1": dat["IMGFUNC1"],
                }
            )

        im = Image.fromarray(np.hstack(slot_imgs)).convert("RGB")
        imgfilename = "piximg_{}.png".format(row["rowsecs"])
        im.save(os.path.join(outdir, imgfilename), "PNG")
        row["img"] = imgfilename
        rows.append(row)
        if len(rows) > max_images:
            break
    return rows


def make_event_reports(bgd_events, outdir=".", redo=False):
    """
    For a table of background events, make a high level summary report
    by obsid and make an individual obsid report for each.

    :param bgd_events: astropy table with times and obsids of events
    :param outdir: output directory
    :param redo: if set, remake plots and reports if they already exist
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    bgd_events = bgd_events[bgd_events["obsid"] != 0]
    bgd_events = bgd_events[bgd_events["obsid"] != -1]

    obs_events = []
    for obsid in np.unique(bgd_events["obsid"]):
        # ignore the obsid == 0 case(s) for now
        if obsid == 0:
            continue

        events = bgd_events[bgd_events["obsid"] == obsid]
        events["index"] = np.arange(len(events))

        slots = {}
        for e in events:
            for s in e["slots"].split(","):
                slots[int(s)] = 1

        # Save the events, the intervals, and some other useful stuff for the per-obsid table
        obs = {
            "events": events.copy(),
            "datestart": DateTime(events[0]["event_tstart"]).date,
            "n_events": len(events),
            "max_dur": np.max(events["duration"]),
            "max_slot_secs": np.max(events["slot_seconds"]),
            "n_slots": len(slots),
            "obsid": obsid,
            "reldir": os.path.join("events", "obs_{:05d}".format(obsid)),
            "dir": os.path.join(outdir, "events", "obs_{:05d}".format(obsid)),
            "pitch": events[0]["pitch"],
        }
        obs_events.append(obs)

    obs_events = sorted(obs_events, key=lambda i: i["datestart"])
    obs_events = obs_events[::-1]

    file_dir = Path(__file__).parent
    template = Template(open(file_dir / "top_level_template.html", "r").read())
    page = template.render(obs_events=obs_events)
    f = open(os.path.join(outdir, "index.html"), "w")
    f.write(page)
    f.close()

    # Do the per-obsid plot and report making
    for obs in obs_events:
        if not os.path.exists(obs["dir"]):
            os.makedirs(obs["dir"])
        else:
            if redo is False:
                continue

        events = []
        for e in obs["events"]:
            event = {k: v for k, v in zip(e.colnames, e.as_void())}
            (event["bgdplot"], event["maxbgd"]) = plot_bgd(event, obs["dir"])
            event["aokalstr"] = plot_aokalstr(event, obs["dir"])
            event["imgrows"] = make_images(
                event["cross_time"] - 100, event["cross_time"] + 300, obs["dir"]
            )
            events.append(event)

        logger.info(f"Making report for {obs['obsid']}")
        obs_template = Template(open(file_dir / "per_obs_template.html", "r").read())
        page = obs_template.render(obsid=obs["obsid"], events=events)
        f = open(os.path.join(obs["dir"], "index.html"), "w")
        f.write(page)
        f.close()


def main():
    """
    Review kadi dwells for new high background events, update a text file table of
    those events, make reports, and notify via email as needed.
    """
    global logger

    opt = get_opt()
    logger = pyyaks.logger.get_logger(level=opt.log_level)

    EVENT_ARCHIVE = os.path.join(opt.data_root, "bgd_events.dat")
    Path(opt.data_root).mkdir(parents=True, exist_ok=True)
    start = None

    bgd_events = []
    if os.path.exists(EVENT_ARCHIVE):
        bgd_events = Table.read(EVENT_ARCHIVE, format="ascii")
    if len(bgd_events) > 0:
        start = DateTime(bgd_events["dwell_datestart"][-1])
        # Remove any bogus events from the real list
        bgd_events = bgd_events[bgd_events["obsid"] != -1]
        bgd_events["slots"] = bgd_events["slots"].astype(str)
        bgd_events["slots_for_sum"] = bgd_events["slots_for_sum"].astype(str)

    # If the user has asked for a start time earlier than the end of the
    # table, delete any rows after the supplied start time
    if opt.start is not None:
        if start is not None:
            if DateTime(opt.start).secs < start.secs:
                bgd_events = bgd_events[
                    bgd_events["dwell_datestart"] < DateTime(opt.start).date
                ]
        start = DateTime(opt.start)
    if start is None:
        start = DateTime(-7)

    new_events, stop = get_events(start)
    if len(new_events) > 0:
        new_events = Table(new_events)
        for obsid in np.unique(new_events["obsid"]):
            if obsid in [0, -1]:
                continue
            url = f"{opt.web_url}/events/obs_{obsid:05d}/index.html"
            logger.warning(f"HI BGD event at in obsid {obsid} {url}")
            if len(opt.emails) > 0:
                send_mail(
                    logger,
                    opt,
                    f"ACA HI BGD event in obsid {obsid}",
                    f"HI BGD in obsid {obsid} report at {url}",
                    __file__,
                )

    if len(bgd_events) > 0:
        bgd_events = vstack([bgd_events, new_events])
    else:
        bgd_events = new_events

    # Add a null event at the end
    bgd_events.add_row()
    bgd_events[-1]["obsid"] = -1
    bgd_events[-1]["dwell_datestart"] = DateTime(stop).date

    bgd_events.write(EVENT_ARCHIVE, format="ascii", overwrite=True)

    make_event_reports(bgd_events, opt.web_out)


if __name__ == "__main__":
    main()
