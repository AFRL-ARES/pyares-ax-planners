import threading
import numpy as np
import pandas as pd

from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    HoverTool,
    NumberFormatter,
    Select,
    TableColumn,
    TabPanel,
    Tabs,
)
from bokeh.palettes import Category10, Category20, Viridis256
from bokeh.plotting import figure, output_file, save
from bokeh.server.server import Server
from pathlib import Path



class ThreadSafeDataStore:

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def update(self, new_data):
        with self._lock:
            self._data = new_data

    def get_copy(self):
        with self._lock:
            return dict(self._data)


class BokehIterativeVisualizer:

    def __init__(self, param_cols, resp_cols, resp_opt_dict, param_bounds):
        self.param_cols = param_cols
        self.resp_cols = resp_cols
        self.resp_opt_dict = resp_opt_dict
        self.param_bounds = param_bounds

        self.store = ThreadSafeDataStore()
        self.active_docs = set()
        self.active_docs_lock = threading.Lock()

        self.param_colors = self._get_color_map(self.param_cols)
        self.resp_colors = self._get_color_map(self.resp_cols)

    def _get_color_map(self, items):
        n = len(items)
        if n <= 10:
            palette = Category10[10]
        elif n <= 20:
            palette = Category20[20]
        else:
            step = max(1, len(Viridis256) // n)
            palette = [Viridis256[i * step] for i in range(n)]
        return {item: palette[i % len(palette)] for i, item in enumerate(items)}

    # -------------------------------------------------------------------------
    # Pareto Front Calculation Logic
    # -------------------------------------------------------------------------
    def _calc_2d_pareto_data(self, data, x_resp, y_resp):
        """Calculates 2D Pareto optimal points considering ONLY x_resp and y_resp."""
        if not data or "iteration" not in data or len(data["iteration"]) == 0:
            return {"x": [], "y": [], "iteration": []}

        x_c, y_c = f"{x_resp}_raw", f"{y_resp}_raw"
        xs = np.array(data[x_c])
        ys = np.array(data[y_c])
        iters = np.array(data["iteration"])

        x_m = xs if self.resp_opt_dict.get(x_resp, True) else -xs
        y_m = ys if self.resp_opt_dict.get(y_resp, True) else -ys

        idx = []
        for i in range(len(xs)):
            dominated = any(
                (x_m[j] <= x_m[i] and y_m[j] <= y_m[i])
                and (x_m[j] < x_m[i] or y_m[j] < y_m[i])
                for j in range(len(xs))
                if j != i
            )
            if not dominated:
                idx.append(i)

        sort_idx = np.argsort(x_m[idx])
        return {
            "x": xs[idx][sort_idx],
            "y": ys[idx][sort_idx],
            "iteration": iters[idx][sort_idx],
        }

    def _calc_nd_pareto_indices(self, data):
        """Calculates non-dominated point indices across ALL response dimensions."""
        if not data or "iteration" not in data or len(data["iteration"]) == 0:
            return []

        # Construct matrix normalized for minimization space
        resps_matrix = []
        for r in self.resp_cols:
            vals = np.array(data[f"{r}_raw"])
            is_min = self.resp_opt_dict.get(r, True)
            resps_matrix.append(vals if is_min else -vals)

        M = np.column_stack(resps_matrix)  # Shape: (N_samples, N_responses)
        n_points = len(M)

        idx = []
        for i in range(n_points):
            dominated = any(
                np.all(M[j] <= M[i]) and np.any(M[j] < M[i])
                for j in range(n_points)
                if j != i
            )
            if not dominated:
                idx.append(i)

        return idx

    def _calc_nd_pareto_data(self, data, x_resp, y_resp):
        """Extracts complete dataset for N-D Pareto points mapped to selected 2D projection."""
        idx = self._calc_nd_pareto_indices(data)

        res = {"x": [], "y": [], "iteration": []}
        for p in self.param_cols:
            res[f"{p}_raw"] = []
        for r in self.resp_cols:
            res[f"{r}_raw"] = []

        if not idx:
            return res

        res["x"] = np.array(data[f"{x_resp}_raw"])[idx]
        res["y"] = np.array(data[f"{y_resp}_raw"])[idx]
        res["iteration"] = np.array(data["iteration"])[idx]

        for p in self.param_cols:
            res[f"{p}_raw"] = np.array(data[f"{p}_raw"])[idx]
        for r in self.resp_cols:
            res[f"{r}_raw"] = np.array(data[f"{r}_raw"])[idx]

        return res

    # -------------------------------------------------------------------------
    # Layout and Visualization Builders
    # -------------------------------------------------------------------------
    @staticmethod
    def style_figure(
        fig, title_size="14pt", label_size="12pt", tick_size="10pt"
    ):
        fig.title.text_font_size = title_size
        fig.title.text_font_style = "bold"

        for axis in [fig.xaxis, fig.yaxis]:
            axis.axis_label_text_font_size = label_size
            axis.axis_label_text_font_style = "bold"
            axis.major_label_text_font_size = tick_size
            axis.major_label_text_font_style = "normal"

    def _build_ui(self, initial_data=None):
        """Constructs an independent tree of Bokeh UI components."""
        data = initial_data or {}

        source = ColumnDataSource(data=data)
        x_resp = self.resp_cols[0]
        y_resp = (
            self.resp_cols[1]
            if len(self.resp_cols) > 1
            else self.resp_cols[0]
        )

        p2d_data = self._calc_2d_pareto_data(data, x_resp, y_resp)
        pnd_data = self._calc_nd_pareto_data(data, x_resp, y_resp)

        pareto_2d_source = ColumnDataSource(data=p2d_data)
        pareto_nd_source = ColumnDataSource(data=pnd_data)

        p1, p2 = self._build_ts_plots(source)
        (
            x_select,
            y_select,
            p_pareto,
            pareto_bg,
            pareto_hover,
            pareto_nd_scatter,
            pareto_table,
        ) = self._build_pareto(
            x_resp, y_resp, source, pareto_2d_source, pareto_nd_source
        )

        ts_layout = column(p1, p2, sizing_mode="stretch_width")
        pareto_layout = column(
            row(x_select, y_select, sizing_mode="stretch_width"),
            p_pareto,
            pareto_table,
            sizing_mode="stretch_width",
        )
        layout = Tabs(
            tabs=[
                TabPanel(child=ts_layout, title="Time Series"),
                TabPanel(child=pareto_layout, title="Pareto Front"),
            ],
            sizing_mode="stretch_width",
        )

        return (
            layout,
            source,
            pareto_2d_source,
            pareto_nd_source,
            x_select,
            y_select,
            p_pareto,
            pareto_bg,
            pareto_hover,
            pareto_nd_scatter,
        )

    def _build_ts_plots(self, source):
        """Creates the time series plot elements."""
        p1 = figure(
            title="Parameters vs. Iteration",
            x_axis_label="Iteration",
            y_axis_label="Normalized Range [0, 1]",
            height=350,
            sizing_mode="stretch_width",
        )
        self.style_figure(
            p1, title_size="14pt", label_size="12pt", tick_size="10pt"
        )
        for p in self.param_cols:
            c = self.param_colors[p]
            p1.line(
                x="iteration",
                y=f"{p}_norm",
                source=source,
                color=c,
                line_width=2,
            )
            r = p1.scatter(
                x="iteration",
                y=f"{p}_norm",
                source=source,
                size=8,
                fill_color=c,
                line_color="black",
                hover_fill_color="#FF007F",
                legend_label=p,
            )
            p1.add_tools(
                HoverTool(
                    renderers=[r],
                    mode="vline",
                    tooltips=[
                        ("Iteration", "@iteration"),
                        ("Parameter", p),
                        ("Actual Value", f"@{p}_raw"),
                    ],
                )
            )

        p2 = figure(
            title="Responses & Best Achieved vs. Iteration",
            x_axis_label="Iteration",
            y_axis_label="Normalized Range [0, 1]",
            x_range=p1.x_range,
            height=350,
            sizing_mode="stretch_width",
        )
        self.style_figure(
            p2, title_size="14pt", label_size="12pt", tick_size="10pt"
        )
        for r in self.resp_cols:
            c = self.resp_colors[r]
            goal = "Min" if self.resp_opt_dict.get(r, True) else "Max"
            r_scat = p2.scatter(
                x="iteration",
                y=f"{r}_norm",
                source=source,
                size=9,
                fill_color=c,
                line_color="darkgray",
                hover_fill_color="#FF007F",
                legend_label=f"{r} (val)",
            )
            p2.line(
                x="iteration",
                y=f"{r}_best_norm",
                source=source,
                color=c,
                line_width=2,
                line_dash="dashed",
            )
            r_best = p2.scatter(
                x="iteration",
                y=f"{r}_best_norm",
                source=source,
                size=7,
                fill_color="white",
                line_color=c,
                hover_fill_color="#FF007F",
                legend_label=f"{r} (best {goal})",
            )
            p2.add_tools(
                HoverTool(
                    renderers=[r_scat, r_best],
                    mode="vline",
                    tooltips=[
                        ("Iteration", "@iteration"),
                        ("Response", r),
                        ("Actual Value", f"@{r}_raw"),
                        ("Best So Far", f"@{r}_best_raw"),
                    ],
                )
            )

        for p in (p1, p2):
            if p.legend:
                legend = p.legend[0]
                p.add_layout(legend, "right")
                legend.click_policy = "hide"

        return (p1, p2)

    def _build_pareto(
        self, x_resp, y_resp, source, pareto_2d_source, pareto_nd_source
    ):
        """Creates Pareto plot elements and DataTable."""
        x_select = Select(
            title="X-Axis Response:", value=x_resp, options=self.resp_cols
        )
        y_select = Select(
            title="Y-Axis Response:", value=y_resp, options=self.resp_cols
        )
        if len(self.resp_cols) <= 2:
            x_select.visible = False
            y_select.visible = False

        p_pareto = figure(
            title="Pareto Front Analysis",
            x_axis_label=f"{x_resp}",
            y_axis_label=f"{y_resp}",
            height=450,
            sizing_mode="stretch_width",
        )
        self.style_figure(
            p_pareto, title_size="14pt", label_size="12pt", tick_size="10pt"
        )

        # 1. Background (All Points)
        pareto_bg = p_pareto.scatter(
            x=f"{x_resp}_raw",
            y=f"{y_resp}_raw",
            source=source,
            color="lightgray",
            size=7,
            hover_fill_color="#FF007F",
            legend_label="All Iterations",
        )

        # 2. 2-Dimensional Pareto Front (Line + Open Scatter)
        p_pareto.line(
            x="x",
            y="y",
            source=pareto_2d_source,
            color="deepskyblue",
            line_width=2,
            line_dash="dashed",
            legend_label="2D Pareto Front",
        )
        p_pareto.scatter(
            x="x",
            y="y",
            source=pareto_2d_source,
            color="deepskyblue",
            size=8,
            line_color="blue",
            legend_label="2D Optimal",
        )

        # 3. N-Dimensional Pareto Front (Gold Stars)
        pareto_nd_scatter = p_pareto.scatter(
            x="x",
            y="y",
            source=pareto_nd_source,
            marker="star",
            size=12,
            fill_color="gold",
            line_color="darkorange",
            legend_label="N-D Pareto Optimal",
        )

        pareto_hover = HoverTool(
            renderers=[pareto_bg, pareto_nd_scatter],
            tooltips=[
                ("Iteration", "@iteration"),
                (x_resp, f"@{x_resp}_raw"),
                (y_resp, f"@{y_resp}_raw"),
            ],
        )
        p_pareto.add_tools(pareto_hover)

        if p_pareto.legend:
            leg = p_pareto.legend[0]
            p_pareto.add_layout(leg, "right")
            leg.click_policy = "hide"

        # 4. DataTable for N-D Pareto Front
        table_columns = [
            TableColumn(field="iteration", title="Iteration", width=60)
        ]
        for p in self.param_cols:
            table_columns.append(
                TableColumn(
                    field=f"{p}_raw",
                    title=f"Param: {p}",
                    formatter=NumberFormatter(format="0.0000"),
                )
            )
        for r in self.resp_cols:
            goal = "Min" if self.resp_opt_dict.get(r, True) else "Max"
            table_columns.append(
                TableColumn(
                    field=f"{r}_raw",
                    title=f"Resp: {r} ({goal})",
                    formatter=NumberFormatter(format="0.0000"),
                )
            )

        pareto_table = DataTable(
            source=pareto_nd_source,
            columns=table_columns,
            height=200,
            sizing_mode="stretch_width",
            index_position=None,
        )

        return (
            x_select,
            y_select,
            p_pareto,
            pareto_bg,
            pareto_hover,
            pareto_nd_scatter,
            pareto_table,
        )

    # -------------------------------------------------------------------------
    # Server Lifecycle & Event Dispatch
    # -------------------------------------------------------------------------
    def bkapp(self, doc):
        """Bokeh server document binding per client."""
        with self.active_docs_lock:
            self.active_docs.add(doc)

        doc.on_session_destroyed(
            lambda session_context: self.active_docs.discard(doc)
        )

        initial_data = self.store.get_copy()
        (
            layout,
            source,
            pareto_2d_source,
            pareto_nd_source,
            x_select,
            y_select,
            p_pareto,
            pareto_bg,
            pareto_hover,
            pareto_nd_scatter,
        ) = self._build_ui(initial_data)

        def update_pareto():
            data = self.store.get_copy()
            pareto_2d_source.data = self._calc_2d_pareto_data(
                data, x_select.value, y_select.value
            )
            pareto_nd_source.data = self._calc_nd_pareto_data(
                data, x_select.value, y_select.value
            )

        def on_dropdown_change(attr, old, new):
            pareto_bg.glyph.x = f"{x_select.value}_raw"
            pareto_bg.glyph.y = f"{y_select.value}_raw"

            p_pareto.xaxis.axis_label = f"{x_select.value}"
            p_pareto.yaxis.axis_label = f"{y_select.value}"

            pareto_hover.tooltips = [
                ("Iteration", "@iteration"),
                (x_select.value, f"@{x_select.value}_raw"),
                (y_select.value, f"@{y_select.value}_raw"),
            ]
            update_pareto()

        x_select.on_change("value", on_dropdown_change)
        y_select.on_change("value", on_dropdown_change)

        def refresh_session_data():
            latest = self.store.get_copy()
            if latest:
                source.data = latest
                update_pareto()

        doc.user_refresh = refresh_session_data
        doc.add_root(layout)

    def save_snapshot(self, filepath: str):
        """Generates an HTML visualization archive alongside full and N-D Pareto CSV exports."""
        current_data = self.store.get_copy()
        snapshot_layout, _, _, _, _, _, _, _, _, _ = self._build_ui(
            current_data
        )

        # Save HTML interface
        output_file(str(Path(filepath)/"visualizer.html"), mode="inline", title="Iteration Archive")
        save(snapshot_layout)

        # Export raw CSV files if data is available
        if (
            current_data
            and "iteration" in current_data
            and len(current_data["iteration"]) > 0
        ):
            # Export 1: Full Iteration Data
            full_df = pd.DataFrame(current_data)
            full_df.to_csv(str(Path(filepath)/"visualizer_full_data.csv"), index=False)

            # Export 2: N-D Pareto Optimal Subset
            nd_indices = self._calc_nd_pareto_indices(current_data)
            pareto_df = full_df.iloc[nd_indices].copy()
            pareto_df.to_csv(str(Path(filepath)/"visualizer_nd_pareto.csv"), index=False)

    def push_update(self, df: pd.DataFrame):
        """Thread-safe update broadcasted across all active documents."""
        data = self._transform_dataframe(df)
        self.store.update(data)

        with self.active_docs_lock:
            docs = list(self.active_docs)

        for d in docs:

            def cb(doc_ref=d):
                try:
                    if hasattr(doc_ref, "user_refresh"):
                        doc_ref.user_refresh()
                except Exception:
                    pass

            d.add_next_tick_callback(cb)

    def _transform_dataframe(self, df: pd.DataFrame) -> dict:
        """Process incoming dataframe into dictionary needed for plots."""
        data = {
            "iteration": (
                df["iteration"].tolist()
                if "iteration" in df.columns
                else list(range(1, len(df) + 1))
            )
        }
        for p in self.param_cols:
            p_min, p_max = self.param_bounds[p]
            data[f"{p}_raw"] = df[p].tolist()
            denom = p_max - p_min
            data[f"{p}_norm"] = (
                ((df[p] - p_min) / denom).tolist()
                if denom != 0
                else [0.5] * len(df)
            )

        for r in self.resp_cols:
            r_raw = df[r]
            data[f"{r}_raw"] = r_raw.tolist()
            r_min, r_max = r_raw.min(), r_raw.max()
            denom = r_max - r_min
            data[f"{r}_norm"] = (
                ((r_raw - r_min) / denom).tolist()
                if denom != 0
                else [0.5] * len(df)
            )
            is_min = self.resp_opt_dict.get(r, True)
            best_raw = r_raw.cummin() if is_min else r_raw.cummax()
            data[f"{r}_best_raw"] = best_raw.tolist()
            data[f"{r}_best_norm"] = (
                ((best_raw - r_min) / denom).tolist()
                if denom != 0
                else [0.5] * len(df)
            )
        return data

class VisualizerServerManager:

    def __init__(self, port=5006):
        self.port = port
        self.current_instance_id = None
        self.server = None
        self.io_thread = None
        self.visualizer = None
        self._lock = threading.Lock()

    def sync_instance(self, instance_id, visualizer_cls, **visualizer_kwargs):
        """Checks if instance_id has changed.

        If changed, cleanly tears down the running server and starts a new
        instance with updated parameter schemas.
        """
        with self._lock:
            # If instance ID matches and server is alive, return current visualizer
            if self.current_instance_id == instance_id and self.server:
                return self.visualizer

            # Teardown active server on instance ID mismatch
            if self.server:
                self._stop_server_unlocked()

            # Instantiate new visualizer with new parameters/meshgrids
            self.visualizer = visualizer_cls(**visualizer_kwargs)

            # Re-bind Bokeh Server on port
            self.server = Server(
                {"/": self.visualizer.bkapp},
                port=self.port,
                allow_websocket_origin=["*"],
            )
            self.server.start()

            # Launch Tornado I/O loop in background thread
            self.io_thread = threading.Thread(
                target=self.server.io_loop.start
            )
            self.io_thread.daemon = True
            self.io_thread.start()

            self.current_instance_id = instance_id
            return self.visualizer

    def _stop_server_unlocked(self):
        """Tears down Tornado I/O loop and unbinds port socket."""
        if self.server:
            # 1. Unbind listening socket
            self.server.stop()

            # 2. Schedule thread-safe stop callback on Tornado loop
            if self.server.io_loop:
                self.server.io_loop.add_callback(self.server.io_loop.stop)

            # 3. Wait for background IO thread to terminate
            if self.io_thread and self.io_thread.is_alive():
                self.io_thread.join(timeout=3.0)

            self.server = None
            self.io_thread = None
            self.visualizer = None

    def push_update(self, df):
        """Pushes data update to active visualizer."""
        with self._lock:
            if self.visualizer:
                self.visualizer.push_update(df)

    def save_snapshot(self, filepath):
        """Saves offline snapshot using active visualizer."""
        with self._lock:
            if self.visualizer:
                self.visualizer.save_snapshot(filepath)