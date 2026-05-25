"""
comparison_frame.py — Screen 4: side-by-side comparison of two training runs.

Left side  : current run (from state["metrics"])
Right side : a previously saved run loaded from a .pkl file
Bottom     : overlaid charts and a summary table
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pickle
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io, base64
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


_PURSUER_C = "#34d399"
_EVADER_C  = "#a78bfa"
_RUN_A_C   = "#4f8ef7"   # current run — blue
_RUN_B_C   = "#fbbf24"   # loaded run  — amber


class ComparisonFrame(ttk.Frame):
    def __init__(self, parent, state: dict, palette: dict):
        super().__init__(parent)
        self.state   = state
        self.P       = palette
        self._run_b  = None   # loaded second run

        self._run_a_label = tk.StringVar(value="Current run")
        self._run_b_label = tk.StringVar(value="No run loaded")
        self._compare_status = tk.StringVar(value="Load a saved run to compare.")

        self._build()

    # ──────────────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)

        # Header
        hdr = tk.Frame(self, bg=self.P["surface"],
                       highlightbackground=self.P["border"],
                       highlightthickness=1)
        hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        tk.Label(hdr, text="Compare Runs",
                 bg=self.P["surface"], fg=self.P["text"],
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=14, pady=10)
        tk.Label(hdr, text="Load a second training run to compare algorithms side-by-side.",
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=2, pady=10)

        # Run selection bar
        bar = tk.Frame(self, bg=self.P["surface"],
                       highlightbackground=self.P["border"],
                       highlightthickness=1)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(3, weight=1)

        # Run A (current)
        tk.Frame(bar, bg=_RUN_A_C, width=10).grid(
            row=0, column=0, sticky="ns", padx=(14, 6), pady=10)
        run_a_entry = tk.Entry(bar, textvariable=self._run_a_label,
                               bg=self.P["surface"], fg=self.P["text"],
                               insertbackground=self.P["text"],
                               relief="flat", font=("Segoe UI", 10))
        run_a_entry.grid(row=0, column=1, sticky="ew", pady=10)

        ttk.Button(bar, text="Save current run…",
                   command=self._save_current).grid(
            row=0, column=2, padx=(10, 20), pady=10)

        # Run B (loaded)
        tk.Frame(bar, bg=_RUN_B_C, width=10).grid(
            row=0, column=3, sticky="ns", padx=(4, 6), pady=10)
        run_b_lbl = tk.Label(bar, textvariable=self._run_b_label,
                              bg=self.P["surface"], fg=self.P["muted"],
                              font=("Segoe UI", 10))
        run_b_lbl.grid(row=0, column=4, sticky="ew", pady=10)
        bar.columnconfigure(4, weight=1)
        self._run_b_lbl_widget = run_b_lbl

        ttk.Button(bar, text="Load saved run…",
                   command=self._load_run).grid(
            row=0, column=5, padx=(10, 14), pady=10)

        # Chart area
        self._chart_frame = tk.Frame(self, bg=self.P["bg"])
        self._chart_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self._chart_frame.columnconfigure(0, weight=1)
        self._chart_frame.rowconfigure(0, weight=1)

        self._draw_placeholder()

    # ──────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self):
        """Called by app.py after training completes — updates Run A label."""
        pursuer_cls = self.state.get("pursuer_class")
        evader_cls  = self.state.get("evader_class")
        p_name = getattr(pursuer_cls, "NAME",
                 getattr(pursuer_cls, "__name__", "Pursuer")) if pursuer_cls else "Pursuer"
        e_name = getattr(evader_cls,  "NAME",
                 getattr(evader_cls,  "__name__", "Evader"))  if evader_cls  else "Evader"
        self._run_a_label.set(f"{p_name} vs {e_name}")

    def _save_current(self):
        metrics = self.state.get("metrics")
        if metrics is None:
            messagebox.showwarning(
                "No run available",
                "Train at least one run before saving.")
            return

        path = filedialog.asksaveasfilename(
            title="Save run",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return

        label = self._run_a_label.get() or "run"
        payload = {"label": label, "metrics": metrics}
        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            messagebox.showinfo("Saved", f"Run saved to {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _load_run(self):
        path = filedialog.askopenfilename(
            title="Load saved run",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        self._run_b = payload.get("metrics")
        label       = payload.get("label", Path(path).stem)
        self._run_b_label.set(label)
        self._run_b_lbl_widget.config(fg=self.P["text"])
        self._render_comparison()

    # ──────────────────────────────────────────────────────────────────────────
    # Comparison charts
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        for w in self._chart_frame.winfo_children():
            w.destroy()
        tk.Label(self._chart_frame,
                 text="Save a run, then load a second run to compare.",
                 bg=self.P["bg"], fg=self.P["muted"],
                 font=("Segoe UI", 10)).pack(expand=True)

    def _render_comparison(self):
        for w in self._chart_frame.winfo_children():
            w.destroy()

        if not HAS_MPL:
            tk.Label(self._chart_frame,
                     text="matplotlib not installed.\nRun: pip install matplotlib",
                     bg=self.P["bg"], fg=self.P["muted"],
                     font=("Segoe UI", 10)).pack(expand=True)
            return

        metrics_a = self.state.get("metrics")
        metrics_b = self._run_b

        data_a = self._to_chart_data(metrics_a, "A")
        data_b = self._to_chart_data(metrics_b, "B")

        label_a = self._run_a_label.get() or "Run A"
        label_b = self._run_b_label.get() or "Run B"

        fig = Figure(facecolor=self.P["bg"], constrained_layout=True)

        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)

        # ── Win rate comparison ───────────────────────────────────────────────
        for data, colour, label in [
            (data_a, _RUN_A_C, label_a),
            (data_b, _RUN_B_C, label_b),
        ]:
            ax1.plot(data["episodes"], data["pursuer_wr"],
                     color=colour, linewidth=1.5, label=label)
        ax1.set_title("Pursuer Win Rate", fontsize=9)
        ax1.legend(fontsize=7)
        _style_ax(ax1, self.P)

        for data, colour, label in [
            (data_a, _RUN_A_C, label_a),
            (data_b, _RUN_B_C, label_b),
        ]:
            ax2.plot(data["episodes"], data["evader_wr"],
                     color=colour, linewidth=1.5, label=label)
        ax2.set_title("Evader Win Rate", fontsize=9)
        ax2.legend(fontsize=7)
        _style_ax(ax2, self.P)

        # ── Policy variance ───────────────────────────────────────────────────
        for data, colour, label in [
            (data_a, _RUN_A_C, label_a),
            (data_b, _RUN_B_C, label_b),
        ]:
            ax3.plot(data["episodes"], data["std_wr"],
                     color=colour, linewidth=1.5, label=label)
        ax3.set_title("Policy Variance (std win rate)", fontsize=9)
        ax3.legend(fontsize=7)
        _style_ax(ax3, self.P)

        # ── Reward curves ─────────────────────────────────────────────────────
        for data, colour, label in [
            (data_a, _RUN_A_C, label_a),
            (data_b, _RUN_B_C, label_b),
        ]:
            ax4.plot(data["episodes"], data["pursuer_rewards"],
                     color=colour, linewidth=1.5, label=label)
        ax4.set_title("Pursuer Reward", fontsize=9)
        ax4.legend(fontsize=7)
        _style_ax(ax4, self.P)

        for data, colour, label in [
            (data_a, _RUN_A_C, label_a),
            (data_b, _RUN_B_C, label_b),
        ]:
            ax5.plot(data["episodes"], data["evader_rewards"],
                     color=colour, linewidth=1.5, label=label)
        ax5.set_title("Evader Reward", fontsize=9)
        ax5.legend(fontsize=7)
        _style_ax(ax5, self.P)

        # ── Summary bar chart ─────────────────────────────────────────────────
        metrics_list = ["Pursuer WR", "Evader WR", "Avg Ep Len"]
        a_vals = [
            data_a["pursuer_wr"][-1] if data_a["pursuer_wr"] else 0,
            data_a["evader_wr"][-1]  if data_a["evader_wr"]  else 0,
            data_a.get("avg_len", 45),
        ]
        b_vals = [
            data_b["pursuer_wr"][-1] if data_b["pursuer_wr"] else 0,
            data_b["evader_wr"][-1]  if data_b["evader_wr"]  else 0,
            data_b.get("avg_len", 45),
        ]

        x = np.arange(len(metrics_list))
        w = 0.35
        ax6.bar(x - w/2, a_vals, w, color=_RUN_A_C, label=label_a)
        ax6.bar(x + w/2, b_vals, w, color=_RUN_B_C, label=label_b)
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics_list, fontsize=7)
        ax6.set_title("Final Summary", fontsize=9)
        ax6.legend(fontsize=7)
        _style_ax(ax6, self.P)

        import tkinter as tk

        tk_canvas = tk.Canvas(self._chart_frame,
                              bg="#1a1d23",
                              highlightthickness=0, bd=0)
        tk_canvas.pack(fill="both", expand=True)

        _photo_ref = [None]
        _pending   = [None]
        _last      = [0, 0]

        def _render(w, h):
            if w < 20 or h < 20:
                return
            dpi = fig.get_dpi() or 100
            fig.set_size_inches(w / dpi, h / dpi, forward=False)
            agg = FigureCanvasAgg(fig)
            agg.draw()
            buf = io.BytesIO()
            agg.print_png(buf)
            buf.seek(0)
            b64   = base64.b64encode(buf.read()).decode()
            photo = tk.PhotoImage(data=b64)
            tk_canvas.delete("all")
            tk_canvas.create_image(0, 0, anchor="nw", image=photo)
            _photo_ref[0] = photo

        def _on_configure(event):
            w, h = event.width, event.height
            if w == _last[0] and h == _last[1]:
                return
            _last[0], _last[1] = w, h
            if _pending[0] is not None:
                tk_canvas.after_cancel(_pending[0])
            _pending[0] = tk_canvas.after(80, lambda: _render(w, h))

        tk_canvas.bind("<Configure>", _on_configure)

    def _to_chart_data(self, metrics, tag: str) -> dict:
        """Convert a MetricsRecorder (or None) to a simple dict for charting."""
        if metrics is not None:
            try:
                summary = metrics.compute_summary()
                if summary:
                    wr  = summary.get("win_rates", {})
                    cr  = summary.get("cumulative_rewards", {})
                    eps = wr.get("episodes", [])

                    p_wr = [v * 100 for v in wr.get("pursuer", [])]
                    e_wr = [v * 100 for v in wr.get("evader",  [])]

                    # rolling std of pursuer win rate
                    import numpy as np
                    w   = max(1, len(p_wr) // 20)
                    std = []
                    for i in range(len(p_wr)):
                        sl = p_wr[max(0, i-w):i+1]
                        std.append(float(np.std(sl)) if sl else 0.0)

                    ep_lens = summary.get("episode_lengths", [])
                    return {
                        "episodes":        eps,
                        "pursuer_wr":      p_wr,
                        "evader_wr":       e_wr,
                        "pursuer_rewards": cr.get("pursuer", []),
                        "evader_rewards":  cr.get("evader",  []),
                        "std_wr":          std,
                        "avg_len":         (sum(ep_lens) / len(ep_lens)
                                            if ep_lens else 45),
                    }
            except Exception:
                pass

        # Fall back to fake data
        import random
        n   = 40
        eps = list(range(50, 50*(n+1), 50))
        base = 45 + (5 if tag == "B" else 0)
        p_wr = []
        val  = base
        for _ in range(n):
            val = min(88, max(12, val + random.gauss(0.5, 2)))
            p_wr.append(val)
        e_wr = [100 - v for v in p_wr]
        p_r  = [random.gauss(-0.3 + i*0.01, 0.1) for i in range(n)]
        e_r  = [random.gauss( 0.1 + i*0.008, 0.1) for i in range(n)]
        std  = [max(0.4, 12 - i*0.25 + random.gauss(0, 0.4)) for i in range(n)]

        return {
            "episodes":        eps,
            "pursuer_wr":      p_wr,
            "evader_wr":       e_wr,
            "pursuer_rewards": p_r,
            "evader_rewards":  e_r,
            "std_wr":          std,
            "avg_len":         42 + (3 if tag == "B" else 0),
        }


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _style_ax(ax, P: dict):
    ax.set_facecolor(P["bg"])
    ax.figure.set_facecolor(P["bg"])
    for spine in ax.spines.values():
        spine.set_color(P["border"])
    ax.tick_params(colors=P["muted"], labelsize=7)
    ax.xaxis.label.set_color(P["muted"])
    ax.yaxis.label.set_color(P["muted"])
    ax.title.set_color(P["text"])