import collections
from collections.abc import Callable

import astropy.units as u
import numpy as np
import plotly.graph_objects as go
from astropy.table import Table
from chandra_aca.aca_image import get_aca_images
from cheta import fetch
from cxotime import CxoTime, CxoTimeLike
from kadi import events
from kadi.commands import get_cmds
from plotly.subplots import make_subplots
from ska_helpers.logging import basic_logger

LOGGER = basic_logger(__name__, level="INFO")
fetch.data_source.set("cxc", "maude allow_subset=False")


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


def plot_dwell(  # noqa: PLR0912, PLR0915 too many statements, too many branches
    start: CxoTimeLike,
    stop: CxoTimeLike,
    dwell_start: CxoTimeLike,
    top_events: Table,
    all_events: Table,
) -> str:
    """
    Generate a plotly plot of backgrounds data and aokalstr over an event.

    This function creates a plot with two or three subplots:
    1. BGDAVG (e-/s) - included if observation has 6x6 data
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
    from aca_hi_bgd.update_bgd_events import get_slot_mags

    slot_mag = get_slot_mags(dwell_start)
    sd_table = get_aca_images(start, stop, source="cxc", bgsub=True)
    slots_data = {slot: sd_table[sd_table["IMGNUM"] == slot] for slot in range(8)}

    # Does the observation have 6x6 data?
    has_6x6 = False
    for slot in range(8):
        if slot in slots_data:
            if np.median(slots_data[slot]["IMGSIZE"]) == 6:
                has_6x6 = True
                break

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

    # Make a figure with 2 or 3 plots for the overall background data for the dwell
    # and the kalman star count.
    if has_6x6:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "BGDAVG (e-/s)",
                "corr background 8th outer min (e-/s)",
                "AOKALSTR",
            ),
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("corr background 8th outer min (e-/s)", "AOKALSTR"),
        )

    # use a fixed number of bins to speed up the plotting if there are a lot of data points
    num_bins = 2000

    # For the corrected outer min, use this as the ymax
    ymax_outermin = 500

    # Add the background traces to the first 1 or 2 plots.
    add_slot_background_traces(
        fig,
        start,
        slot_mag,
        slots_data,
        has_6x6,
        colors,
        num_bins,
        ymax_outermin,
    )

    # Add the range of the events to the first two plots as a shaded region
    add_event_shade_regions(fig, "grey", start, all_events, has_6x6, ymax_outermin)
    add_event_shade_regions(fig, "red", start, top_events, has_6x6, ymax_outermin)

    # Add aokalstr data to plot
    add_aokalstr_trace(fig, start, stop, has_6x6, num_bins)

    fig.update_xaxes(matches="x")  # Define the layout

    # Finish up by labeling the plots and setting the ranges
    if has_6x6:
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
    else:
        fig.update_layout(
            xaxis_title="Obs Time (ks)",
            yaxis_title="corr background 8th outer min",
            xaxis2_title="Obs Time (ks)",
            yaxis2_title="AOKALSTR (N)",
            yaxis={"range": [0, 500]},
            yaxis2={"range": [0, 8]},
            height=500,
            width=1000,
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


def add_aokalstr_trace(fig, start, stop, has_6x6, num_bins):
    """
    Add the AOKALSTR data to the plot.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to add the trace to.
    start : CxoTimeLike
        Start time of the dwell.
    stop : CxoTimeLike
        Stop time of the dwell.
    has_6x6 : bool
        Whether the slot has mostly 6x6 data.
    num_bins : int
        Number of bins to use for the data.

    Returns
    -------
    None
        The function modifies the figure in place.
    """
    aokalstr = fetch.Msid("AOKALSTR", CxoTime(start).secs, CxoTime(stop).secs)
    values = np.array(aokalstr.vals).astype(int)
    dtimes = (aokalstr.times - aokalstr.times[0]) / 1000.0
    if len(dtimes) > num_bins:
        a_times, a_data = rebin_data(dtimes, values, num_bins, np.min)
        dtimes = a_times
        values = a_data

    fig.add_trace(
        go.Scatter(
            x=dtimes, y=values, mode="lines", name="aokalstr", line={"color": "blue"}
        ),
        row=1,
        col=3 if has_6x6 else 2,
    )


def add_event_shade_regions(fig, color, start, events, has_6x6, max_y):
    """
    Add event shaded regions to the background trace plots.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to add the shaded regions to.
    start : CxoTimeLike
        Start time of the dwell.
    events : astropy.table.Table
        Table of events to be plotted.
    has_6x6 : bool
        Whether the slot has mostly 6x6 data.

    Returns
    -------
    None
        The function modifies the figure in place.

    """
    for event in events:
        # Add a rectangle to the plot for each event
        fig.add_shape(
            type="rect",
            x0=(event["tstart"] - CxoTime(start).secs) / 1000.0,
            y0=0,
            x1=(event["tstop"] - CxoTime(start).secs) / 1000.0,
            y1=max_y,
            fillcolor=color,
            opacity=0.5,
            line_width=0,
            row=1,
            col=2 if has_6x6 else 1,
        )


def add_slot_background_traces(
    fig,
    start,
    slot_mag,
    slots_data,
    has_6x6,
    colors,
    num_bins,
    ymax_outermin=500,
):
    """
    Add the background traces to the plot.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to add the traces to.
    start : CxoTimeLike
        Start time of the dwell.
    slot_mag : dict
        Dictionary of slot magnitudes keyed by slot.
    slots_data : dict
        Dictionary of slot data tables with keys 0-7 and tables with columns
        "TIME", "IMGFUNC", "IMGSIZE", "outer_min_7_magsub", "bgdavg_es", "threshold".
    has_6x6 : bool
        Whether the slot has mostly 6x6 data.
    colors : list
        List of colors for the traces.
    num_bins : int
        Number of bins to use for the data.
    ymax_outermin : int
        Maximum value for the outer minimum plot.

    Returns
    -------
    None
        The function modifies the figure in place.

    """
    from aca_hi_bgd.update_bgd_events import get_background_data_and_thresh

    for slot in range(8):
        slot_data = slots_data[slot]

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

        if has_6x6:
            fig.add_trace(
                go.Scatter(
                    x=dtimes,
                    y=y1_data,
                    mode="markers",
                    marker={"color": colors[slot]},
                    legendgroup=f"Slot {slot}",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=dtimes,
                y=y2_data,
                mode="markers",
                name=f"Slot {slot}",
                marker={"color": colors[slot]},
                legendgroup=f"Slot {slot}",
                showlegend=True,
            ),
            row=1,
            col=2 if has_6x6 else 1,
        )

        # Add a marker at just below the max value for each point beyond the max value
        high_ok = y2_data > ymax_outermin
        hovertext = [f"slot {slot}, yval {int(y)}" for y in y2_data[high_ok]]
        fig.add_trace(
            go.Scatter(
                x=dtimes[high_ok],
                y=ymax_outermin * np.ones_like(dtimes[high_ok]) - 20,
                mode="markers",
                marker={"color": "black", "symbol": "arrow-bar-up", "size": 10},
                legendgroup=f"Slot {slot}",
                showlegend=False,
                hoverinfo="text",
                hovertext=hovertext,
            ),
            row=1,
            col=2 if has_6x6 else 1,
        )

        # Add horizontal line for the threshold in a subplot
        threshold = np.median(slot_data["threshold"][ok])
        if np.median(slot_data["IMGSIZE"][ok]) == 6:
            figure_col = 1
        elif np.median(slot_data["IMGSIZE"][ok]) == 8:
            figure_col = 2 if has_6x6 else 1
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
    from aca_hi_bgd.update_bgd_events import (
        get_background_data_and_thresh,
        get_slot_mags,
    )

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
    imagesizes = collections.defaultdict(list)

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
                imagesizes[slot].append(slot_data["IMGSIZE"][idx])
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
                imagesizes[slot].append(8)
                pixvals = np.zeros((sz, sz))
                np.fill_diagonal(pixvals, 255)

            img = np.zeros((10, 10))
            img[1:9, 1:9] = pixvals.transpose()
            slot_stack.append(img)
        image_stacks.append(slot_stack)
    return times, image_stacks, bgdavgs, outer_mins, imagesizes


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
    start = CxoTime(start) - 10 * u.s
    stop = CxoTime(stop) + 10 * u.s

    # if stop - start > 300 seconds, clip it
    if (stop - start).to(u.s).value > 300:
        stop = start + 300 * u.s

    times, image_stacks, bgdavgs, outer_mins, imagesizes = get_images_for_plot(
        start, stop
    )

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
    set_animation_layout(num_frames, COLORMAP, fig)

    # Hide axis tick labels, grids, and zero lines for all subplots
    customize_animation_layout_axes(fig, num_stacks)

    # Adjust the y-position of the labels
    label_y_position = -0.4  # Move closer to the images

    # Create frames for the animation
    frames = make_animation_frames(
        times,
        image_stacks,
        bgdavgs,
        outer_mins,
        imagesizes,
        num_stacks,
        num_frames,
        COLORMAP,
        label_y_position,
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


def make_animation_frames(
    times,
    image_stacks,
    bgdavgs,
    outer_mins,
    imagesizes,
    num_stacks,
    num_frames,
    COLORMAP,
    label_y_position,
):
    """
    Create frames for the animation.

    Parameters
    ----------
    times : numpy.ndarray
        Array of times for the frames.
    image_stacks : list
        List of image stacks for each slot.
    bgdavgs : dict
        Dictionary with slot numbers as keys and lists of background averages.
    outer_mins : dict
        Dictionary with slot numbers as keys and lists of outer minimum values.
    imagesizes : dict
        Dictionary with slot numbers as keys and lists of image sizes.
    num_stacks : int
        Number of stacks (slots).
    num_frames : int
        Number of frames in the animation.
    COLORMAP : str
        Color map for the heatmap.
    label_y_position : float
        Y-position for the labels in the animation.

    Returns
    -------
    list
        List of frames for the animation.
    """
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
        new_titles = []
        for i in range(num_stacks):
            if imagesizes[i][frame_idx] == 6:
                new_titles.append(f"bgdavg: {bgdavgs[i][frame_idx]:.0f}")
            else:
                new_titles.append(f"outermin: {outer_mins[i][frame_idx]:.0f}")

        annotations = [
            {
                "x": 0,
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

    return frames


def customize_animation_layout_axes(fig, num_stacks):
    """
    Customize the layout of the animation axes.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to customize.
    num_stacks : int
        Number of stacks (subplots).

    Returns
    -------
    None
        The function modifies the figure in place.


    """
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


def set_animation_layout(num_frames, COLORMAP, fig):
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


def plot_events_manvr(bgd_events):
    fig = go.Figure()

    # refetch the maneuver angles for now
    man_angles = []
    frac_years = []
    hovertext = []
    for d in bgd_events:
        manvrs = events.manvrs.filter(kalman_start__exact=d["dwell_datestart"])
        if len(manvrs) > 0:
            man_angles.append(manvrs[0].angle)
            frac_years.append(CxoTime(d["dwell_datestart"]).frac_year)
            hovertext.append(f"obs{d['obsid']:05d}")
        else:
            continue

    fig.add_trace(
        go.Scatter(
            x=frac_years,
            y=man_angles,
            hoverinfo="text",
            hovertext=hovertext,
            mode="markers",
        )
    )
    fig.update_layout(
        yaxis_title="Maneuver angle (deg)",
        height=400,
        width=600,
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True},
    )


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

    # Get the dates of the events
    frac_years = np.array(
        [CxoTime(d["dwell_datestart"]).frac_year for d in dwell_events]
    )

    # Put these in year bins, but bin up so now is the end of the last year bin.
    earliest = frac_years.min()
    now_frac_year = CxoTime.now().frac_year
    n_years = int(np.ceil(now_frac_year - earliest))
    bins = np.arange(now_frac_year - n_years, now_frac_year + 1, 1.0)

    hist, _ = np.histogram(frac_years, bins=bins)
    # get the bin centers for the plot
    x = bins[:-1] + 0.5
    # set the hovertext to include the count and the start/stop of the bin
    hovertext = [
        f"{y} events from {x0:.1f}:{x1:.1f}"
        for x0, x1, y in zip(bins[:-1], bins[1:], hist, strict=False)
    ]

    fig.add_trace(
        go.Bar(x=x, y=hist, name="Events", hovertext=hovertext, hoverinfo="text")
    )
    fig.update_layout(
        title="Number of ACA HI BGD events per year",
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


def plot_events_rel_perigee(bgd_events: Table) -> str:
    frac_year = [CxoTime(d["dwell_datestart"]).frac_year for d in bgd_events]

    dtimes_perigee = []
    for d in bgd_events:
        # Get orbit events for a week before and after the dwell start
        cmds = get_cmds(
            start=CxoTime(d["tstart"]) - 7 * u.day,
            stop=CxoTime(d["tstart"]) + 7 * u.day,
            type="ORBPOINT",
            event_type="EPERIGEE",
        )
        # Find the time of the closest perigee
        if len(cmds) > 0:
            # Get the delta time to all perigee events
            dtime_peri = CxoTime(d["tstart"]).secs - cmds["time"]

            # Find the closest perigee event
            idx = np.argmin(np.abs(dtime_peri))

            # save the delta time
            dtimes_perigee.append(dtime_peri[idx])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frac_year,
            y=dtimes_perigee,
            mode="markers",
            hoverinfo="text",
            hovertext=[f"obs{d['obsid']:05d}" for d in bgd_events],
        )
    )
    fig.update_layout(
        title="Event start time relative to perigee",
        yaxis_title="event time - perigee (s)",
        height=400,
        width=600,
        yaxis={"range": [-150000, 150000]},
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True},
    )


def plot_events_delay(dwell_events: list) -> str:
    """
    Plot delay between dwell starts and event starts.

    Make a plotly scatter plot of the time from the start of the dwell to the
    time of the event for each high background event.

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

    # Get the delay values and the dates
    # For each event, the "delay" is the time between the dwell start and the first event start
    delays = [
        CxoTime(d["first_event_start"]).secs - CxoTime(d["dwell_datestart"]).secs
        for d in dwell_events
    ]
    frac_year = [CxoTime(d["dwell_datestart"]).frac_year for d in dwell_events]
    hovertext = [f"obs{d['obsid']}" for d in dwell_events]

    fig.add_trace(
        go.Scatter(
            x=frac_year,
            y=delays,
            mode="markers",
            hovertext=hovertext,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Event start time relative to dwell start",
        yaxis_title="event start - dwell start (s)",
        height=400,
        width=600,
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True},
    )


def plot_events_pitch(dwell_events: list) -> str:
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
    hovertext = [f"obs{d['obsid']:05d}" for d in dwell_events]

    fig.add_trace(
        go.Scatter(
            x=frac_year,
            y=pitches,
            hovertext=hovertext,
            hoverinfo="text",
            mode="markers",
        )
    )

    fig.update_layout(
        title="Pitch of ACA HI BGD events over time",
        yaxis_title="Pitch",
        height=400,
        width=600,
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True},
    )
