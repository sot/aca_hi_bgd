import os

import argparse
import numpy as np
from pathlib import Path
import smtplib
from email.message import EmailMessage
from astropy import table
from scipy.signal import medfilt
import imghdr
import matplotlib.pyplot as plt
from itertools import count

from Ska.Matplotlib import plot_cxctime
import pyyaks
from cxotime import CxoTime
from cheta import fetch
from cheta.utils import logical_intervals
from kadi.commands import get_observations
from kadi import events
import maude

# %%
fetch.data_source.set('maude allow_subset=False')

# %%
msid_prefix = 'A'
image_function_pea = [f'{msid_prefix}AIMGF{i}1' for i in range(8)]
background_avg_msids = [f'{msid_prefix}CA00110', f'{msid_prefix}CA00326',
                        f'{msid_prefix}CA00542', f'{msid_prefix}CA00758',
                        f'{msid_prefix}CA00974', f'{msid_prefix}CA01190',
                        f'{msid_prefix}CA01406', f'{msid_prefix}CA01622']


# %%
def get_opt(args=None):
    parser = argparse.ArgumentParser(description="MAUDE High Background event finder")
    parser.add_argument("--start",
                        help='Start date')
    parser.add_argument("--obsid",
                        help="Run on individual obsid for testing",
                        type=int)
    parser.add_argument("--data-root",
                        default="./data",
                        help="Output data directory")
    parser.add_argument("--email",
                        action='append',
                        dest='emails',
                        default=['aca@cfa.harvard.edu'],
                        help="Email address for notificaion")
    args = parser.parse_args()
    return args


# %%
def merge_date_intervals(intervals, t_fuzz=20):
    tab = table.Table(intervals, copy=False)
    tab.sort('tstart')
    tab_dict = [dict(zip(tab.colnames, row)) for row in tab]
    merged = [tab_dict[0]]
    for i, row in enumerate(tab_dict[1:]):
        if row['tstart'] <= (merged[-1]['tstop'] + t_fuzz):
            merged[-1]['tstop'] = row['tstop']
        else:
            merged.append(row)
    for entry in merged:
        entry['duration'] = entry['tstop'] - entry['tstart']
    return table.Table(merged)


# %%
def warn_email(warn_obs, opt, image_file):
    recipients = opt.emails
    me = os.environ['USER'] + '@head.cfa.harvard.edu'
    msg = EmailMessage()
    msg['Subject'] = f"MAUDE data hi bgd obsid {warn_obs['obsid']}"
    msg['From'] = me
    msg['To'] = ','.join(recipients)

    with open(image_file, 'rb') as fp:
        img_data = fp.read()
        msg.add_attachment(img_data, maintype='image',
                           subtype=imghdr.what(None, img_data))

    s = smtplib.SMTP('localhost')
    s.sendmail(me, recipients, msg.as_string())
    s.quit()


# %%
def get_obs_intervals(telem):
    """
    Get intervals of high background
    """
    cnts_thresh = 200
    time_thresh = 13 * 1.025
    intervals = []
    for trak, bgd in zip(image_function_pea, background_avg_msids):
        trak_ok = telem[trak].vals == 'TRAK'
        if np.count_nonzero(trak_ok) < 2:
            return []
        msid_medfilt_vals = medfilt(telem[bgd].vals[trak_ok])
        msid_medfilt_times = telem[bgd].times[trak_ok]
        interp_vals = np.interp(telem['AOACASEQ'].times,
                                msid_medfilt_times,
                                msid_medfilt_vals)
        ok = ((interp_vals > cnts_thresh)
              & (telem['AOACASEQ'].vals == 'KALM')
              & (telem['AOPCADMD'].vals == 'NPNT'))
        slot_intervals = logical_intervals(
            telem['AOACASEQ'].times,
            ok)
        slot_intervals = slot_intervals[slot_intervals['duration'] > time_thresh]
        if len(slot_intervals) > 0:
            intervals.append(slot_intervals)

    if len(intervals) == 0:
        return []
    event_list = table.vstack(intervals)
    merged = merge_date_intervals(event_list)
    return merged


def make_plot(warn_row, telem, image_file):
    plt.figure()
    for i, trak, bgd in zip(count(), image_function_pea, background_avg_msids):
        trak_ok = telem[trak].vals == 'TRAK'
        msid_medfilt_vals = medfilt(telem[bgd].vals[trak_ok])
        msid_medfilt_times = telem[bgd].times[trak_ok]
        interp_vals = np.interp(telem['AOACASEQ'].times,
                                msid_medfilt_times,
                                msid_medfilt_vals)
        plot_cxctime(telem['AOACASEQ'].times, interp_vals, '.',
                     markersize=2.5,
                     label=bgd)
    plt.ylabel("medfilt avg bgd")
    plt.grid()
    plt.legend()
    plt.savefig(image_file)


# %%
def main():
    global logger

    opt = get_opt()
    logger = pyyaks.logger.get_logger(level=pyyaks.logger.INFO)

    if not Path(opt.data_root).exists():
        Path(opt.data_root).mkdir(parents=True)

    now = CxoTime.now()
    # Get DSN comms within a day.
    dsn_comms = events.dsn_comms.filter(start=now - 1,
                                        stop=now + 1)

    # If we're in comm or just out of comm (an hour), don't do anything.
    for comm in dsn_comms:
        if (now.secs > (comm.tstart)) & (now.secs < (comm.tstop + (60 * 60))):
            logger.info(f"Within 1 hour of COMM {comm}. Exiting.")
            return None

    warn_file = Path(opt.data_root) / 'maude_warn.dat'
    WARNED = []
    if warn_file.exists():
        WARNED = table.Table.read(warn_file, format='ascii')

    last_time_file = Path(opt.data_root) / 'maude_check.dat'
    LAST_TIME = None
    if last_time_file.exists():
        LAST_TIME = table.Table.read(last_time_file, format='ascii')[0]['time']
    else:
        if opt.start is not None:
            LAST_TIME = CxoTime(opt.start)
        else:
            LAST_TIME = (CxoTime.now() - 40)

    vcdu_count = maude.get_msids('STAT_5MIN_COUNT_CCSDSVCD')
    time_last_backorbit = CxoTime(vcdu_count['data'][0]['times'][0])

    if opt.obsid:
        obss = get_observations(obsid=opt.obsid)
        logger.info(f"Attempting to process on obsid {opt.obsid}")
    else:
        obss = get_observations(start=LAST_TIME)
        logger.info(f"Checking {len(obss)} obs from {CxoTime(LAST_TIME).date}")

    # For each obsid, if it is a science observation warn if it has background
    # events.  Save date of last complete telemetry that is checked.
    for obs in obss:
        tstart = CxoTime(obs['obs_start']).secs
        tstop = CxoTime(obs['obs_stop']).secs

        # Skip ERs
        if obs['obsid'] >= 38000:
            continue

        # If the observation extends into the realtime data, stop.
        if CxoTime(obs['obs_stop']).secs > (time_last_backorbit.secs - (20 * 60)):
            logger.info(f"Hit last backorbit in {obs['obsid']}")
            break

        # Get telemetry
        msids = (['AOPCADMD'] + ['AOACASEQ'] +
                 background_avg_msids + image_function_pea)
        telem = fetch.MSIDset(msids, tstart, tstop)
        max_times = [telem[msid].times[-1] for msid in msids]
        if np.min(max_times) < (tstop - 100):
            logger.warn(f"Telem incomplete for {obs['obsid']}")
            continue
        if tstop > LAST_TIME:
            LAST_TIME = tstop
            table.Table([{'time': LAST_TIME,
                        'date': CxoTime(LAST_TIME).date}]).write(
                            last_time_file, format='ascii')
            logger.info(f"Updating {last_time_file} with time {LAST_TIME}.date")

        # Look for high background intervals
        obs_intervals = get_obs_intervals(telem)

        if len(obs_intervals) > 0:
            warn_row = {'obsid': obs['obsid'],
                        'start': obs['obs_start'],
                        'stop': obs['obs_stop']}
            image_file = Path(opt.data_root) / f"aca_bgd_obsid_{obs['obsid']}.png"
            make_plot(warn_row, telem, image_file)
            logger.info("Sending warn email for {obs['obsid']}")
            warn_email(warn_row, opt, image_file)
            if len(WARNED) == 0:
                WARNED = table.Table([warn_row])
            else:
                WARNED.add_row(warn_row)
            WARNED.write(warn_file, format='ascii')
        else:
            logger.debug(f"No high background found in {obs['obsid']}")


if __name__ == '__main__':
    main()
