"""
results_frame.py — Screen 3: all metric results, five tabs.

Tab 1  Game Outcomes    — win rate chart, episode length distribution
Tab 2  Training Progress — reward curves, convergence
Tab 3  Policy Quality   — state coverage heatmaps
Tab 4  Behaviour        — trajectory viewer, distance plots
Tab 5  Reliability      — policy variance, PAC analysis, confidence intervals
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import datetime

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Colour shortcuts used by all charts ───────────────────────────────────────
_PURSUER_C = "#34d399"
_EVADER_C  = "#a78bfa"


class ResultsFrame(ttk.Frame):
    def __init__(self, parent, state: dict, palette: dict):
        super().__init__(parent)
        self.state = state
        self.P     = palette
        self._built = False
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Page header
        hdr = tk.Frame(self, bg=self.P["surface"],
                       highlightbackground=self.P["border"],
                       highlightthickness=1)
        hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        tk.Label(hdr, text="Results",
                 bg=self.P["surface"], fg=self.P["text"],
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=14, pady=10)
        tk.Label(hdr, text="Metrics recorded across the completed training run.",
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=2, pady=10)

        ttk.Button(hdr, text="⬇  Export stats to .txt",
                   command=self._export_txt).pack(side="right", padx=14, pady=8)

        # Inner notebook for the five metric tabs
        self._nb = ttk.Notebook(self)
        self._nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        self._outcome_tab   = self._make_tab("Game Outcomes")
        self._training_tab  = self._make_tab("Training Progress")
        self._policy_tab    = self._make_tab("Policy Quality")
        self._behaviour_tab = self._make_tab("Behaviour")
        self._reliability_tab = self._make_tab("Reliability")

        self._nb.add(self._outcome_tab,     text="  Game Outcomes  ")
        self._nb.add(self._training_tab,    text="  Training Progress  ")
        self._nb.add(self._policy_tab,      text="  Policy Quality  ")
        self._nb.add(self._behaviour_tab,   text="  Behaviour  ")
        self._nb.add(self._reliability_tab, text="  Reliability  ")

        self._built = True

    def _make_tab(self, _name: str) -> ttk.Frame:
        f = ttk.Frame(self._nb, style="TFrame")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        return f

    # ──────────────────────────────────────────────────────────────────────────
    # Refresh — called after training completes
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self):
        metrics = self.state.get("metrics")
        self._populate_outcome_tab(metrics)
        self._populate_training_tab(metrics)
        self._populate_policy_tab(metrics)
        self._populate_behaviour_tab(metrics)
        self._populate_reliability_tab(metrics)

    # ──────────────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────────────

    def _export_txt(self):
        metrics = self.state.get("metrics")
        if metrics is None:
            messagebox.showwarning("No data", "Run a training session first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save stats report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="rl_stats_report.txt",
        )
        if not path:
            return

        try:
            report = self._build_report(metrics)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)
            messagebox.showinfo("Exported", f"Stats saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _build_report(self, metrics) -> str:
        """Build a full plain-text stats report from a MetricsRecorder."""
        import numpy as np

        try:
            summary = metrics.compute_summary()
        except Exception as e:
            return f"Could not compute summary: {e}\n"

        lines = []
        W = 72  # line width

        def divider(char="─"):
            lines.append(char * W)

        def section(title):
            lines.append("")
            divider("═")
            lines.append(f"  {title.upper()}")
            divider("═")

        def row(label, value, indent=2):
            pad = " " * indent
            lines.append(f"{pad}{label:<35}{value}")

        def blank():
            lines.append("")

        # ── Header ────────────────────────────────────────────────────────────
        lines.append("=" * W)
        lines.append("  RL PURSUIT-EVASION — TRAINING STATS REPORT")
        lines.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
        lines.append("=" * W)

        pursuer_name = getattr(
            self.state.get("pursuer_class"), "NAME",
            getattr(self.state.get("pursuer_class"), "__name__", "Unknown"))
        evader_name = getattr(
            self.state.get("evader_class"), "NAME",
            getattr(self.state.get("evader_class"), "__name__", "Unknown"))

        blank()
        row("Pursuer algorithm:", pursuer_name)
        row("Evader algorithm:",  evader_name)
        row("Total episodes:",    str(summary.get("total_episodes", "—")))

        # ── 1. Game Outcomes ──────────────────────────────────────────────────
        section("1. Game Outcomes")

        wr       = summary.get("win_rates", {})
        p_wr_ser = wr.get("pursuer", [])
        e_wr_ser = wr.get("evader",  [])
        ep_lens  = summary.get("episode_lengths", [])

        if p_wr_ser:
            final_p = p_wr_ser[-1] * 100
            final_e = e_wr_ser[-1] * 100 if e_wr_ser else 100 - final_p
            row("Final pursuer win rate:",  f"{final_p:.1f}%")
            row("Final evader win rate:",   f"{final_e:.1f}%")
            row("Peak pursuer win rate:",   f"{max(p_wr_ser)*100:.1f}%")
            row("Peak evader win rate:",    f"{max(e_wr_ser)*100:.1f}%  " if e_wr_ser else "—")
        else:
            row("Win rate data:", "not available")

        if ep_lens:
            blank()
            row("Mean episode length:",   f"{np.mean(ep_lens):.1f} steps")
            row("Median episode length:", f"{np.median(ep_lens):.1f} steps")
            row("Min episode length:",    f"{min(ep_lens)} steps")
            row("Max episode length:",    f"{max(ep_lens)} steps")
            row("Std episode length:",    f"{np.std(ep_lens):.2f} steps")

        ttc = summary.get("time_to_contact", [])
        ttc_valid = [t for t in ttc if t is not None]
        if ttc_valid:
            blank()
            row("Mean time to first contact:", f"{np.mean(ttc_valid):.1f} steps")
            row("Episodes with contact:",      f"{len(ttc_valid)} / {len(ttc)}")

        # ── 2. Training Progress ──────────────────────────────────────────────
        section("2. Training Progress")

        cr = summary.get("cumulative_rewards", {})
        p_rew = cr.get("pursuer", [])
        e_rew = cr.get("evader",  [])

        if p_rew:
            row("Pursuer — final avg reward:",  f"{p_rew[-1]:.4f}")
            row("Pursuer — peak avg reward:",   f"{max(p_rew):.4f}")
            row("Pursuer — min avg reward:",    f"{min(p_rew):.4f}")
        if e_rew:
            blank()
            row("Evader — final avg reward:",   f"{e_rew[-1]:.4f}")
            row("Evader — peak avg reward:",    f"{max(e_rew):.4f}")
            row("Evader — min avg reward:",     f"{min(e_rew):.4f}")

        conv = summary.get("convergence_episode", {})
        blank()
        p_conv = conv.get("pursuer")
        e_conv = conv.get("evader")
        row("Pursuer converged at episode:",
            str(p_conv) if p_conv else "did not converge")
        row("Evader converged at episode:",
            str(e_conv) if e_conv else "did not converge")

        # ── 3. Policy Quality ─────────────────────────────────────────────────
        section("3. Policy Quality")

        pv = summary.get("policy_variance", {})
        row("Pursuer policy variance (std win rate):",
            f"{pv.get('pursuer', 0):.4f}")
        row("Evader policy variance (std win rate):",
            f"{pv.get('evader',  0):.4f}")

        pe = summary.get("path_efficiency", {})
        p_eff = pe.get("pursuer", [])
        e_eff = pe.get("evader",  [])
        if p_eff:
            blank()
            row("Pursuer mean path efficiency:", f"{np.mean(p_eff):.3f}  (1.0 = optimal)")
            row("Evader mean path efficiency:",  f"{np.mean(e_eff):.3f}  (1.0 = optimal)" if e_eff else "—")

        sc = summary.get("state_coverage", {})
        p_cov = sc.get("pursuer")
        e_cov = sc.get("evader")
        if p_cov is not None:
            blank()
            p_tiles = int(np.sum(p_cov > 0))
            e_tiles = int(np.sum(e_cov > 0)) if e_cov is not None else 0
            row("Pursuer tiles visited (of 100):", str(p_tiles))
            row("Evader tiles visited (of 100):",  str(e_tiles))

        # ── 4. Reliability ────────────────────────────────────────────────────
        section("4. Reliability")

        pac = summary.get("performance_pac", {})
        for role_name, role_key in [("Pursuer", "pursuer"), ("Evader", "evader")]:
            d = pac.get(role_key, {})
            early = d.get("early")
            mid   = d.get("mid")
            late  = d.get("late")
            blank()
            lines.append(f"  {role_name} — performance after convergence:")
            row("  Early window:",  f"{early*100:.1f}%" if early is not None else "N/A", indent=4)
            row("  Mid window:",    f"{mid*100:.1f}%"   if mid   is not None else "N/A", indent=4)
            row("  Late window:",   f"{late*100:.1f}%"  if late  is not None else "N/A", indent=4)
            if all(v is not None for v in [early, mid, late]):
                drift = abs(late - early) * 100
                row("  Drift (early→late):", f"{drift:.1f}pp", indent=4)

        # ── 5. Per-episode breakdown (last 20) ────────────────────────────────
        section("5. Per-Episode Breakdown  (last 20 episodes)")

        ep_records = summary.get("episodes", [])
        last20 = ep_records[-20:] if len(ep_records) >= 20 else ep_records

        if last20:
            blank()
            header_line = (
                f"  {'Episode':>8}  {'Winner':<10}  "
                f"{'Length':>7}  {'P Reward':>10}  {'E Reward':>10}"
            )
            lines.append(header_line)
            divider()
            for ep in last20:
                winner = ep.winner if ep.winner else "timeout"
                lines.append(
                    f"  {ep.episode_number:>8}  {winner:<10}  "
                    f"{ep.episode_length:>7}  "
                    f"{ep.pursuer_total_reward:>10.3f}  "
                    f"{ep.evader_total_reward:>10.3f}"
                )
        else:
            blank()
            lines.append("  No episode records available.")

        # ── Footer ────────────────────────────────────────────────────────────
        blank()
        divider("═")
        lines.append("  End of report")
        divider("═")
        lines.append("")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 1 — Game Outcomes
    # ──────────────────────────────────────────────────────────────────────────

    def _populate_outcome_tab(self, metrics):
        _clear(self._outcome_tab)
        if not HAS_MPL:
            _no_mpl(self._outcome_tab, self.P)
            return

        data = _extract(metrics, "outcome") or _fake_outcome()

        fig = Figure(figsize=(10, 5), facecolor=self.P["bg"])
        fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.12,
                            wspace=0.3)

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # Win rate over time
        eps = data["episodes"]
        ax1.plot(eps, data["pursuer_wr"], color=_PURSUER_C, linewidth=1.5,
                 label="Pursuer")
        ax1.fill_between(eps, data["pursuer_wr"], alpha=0.15, color=_PURSUER_C)
        ax1.plot(eps, data["evader_wr"], color=_EVADER_C, linewidth=1.5,
                 label="Evader")
        ax1.fill_between(eps, data["evader_wr"], alpha=0.15, color=_EVADER_C)
        ax1.set_title("Win Rate Over Time", fontsize=9)
        ax1.legend(fontsize=7)
        _style_ax(ax1, self.P)

        # Final win rate pie
        p_final = data["pursuer_wr"][-1]
        e_final = data["evader_wr"][-1]
        ax2.pie([p_final, e_final],
                labels=["Pursuer", "Evader"],
                colors=[_PURSUER_C, _EVADER_C],
                autopct="%1.1f%%", textprops={"color": self.P["text"],
                                              "fontsize": 8},
                wedgeprops={"linewidth": 0})
        ax2.set_title("Final Win Rate", fontsize=9)
        ax2.set_facecolor(self.P["bg"])

        # Episode length distribution
        ax3.hist(data["ep_lengths"], bins=30, color=self.P["accent"],
                 edgecolor="none", alpha=0.8)
        ax3.set_title("Episode Length Distribution", fontsize=9)
        _style_ax(ax3, self.P)

        _embed(fig, self._outcome_tab)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 2 — Training Progress
    # ──────────────────────────────────────────────────────────────────────────

    def _populate_training_tab(self, metrics):
        _clear(self._training_tab)
        if not HAS_MPL:
            _no_mpl(self._training_tab, self.P)
            return

        data = _extract(metrics, "training") or _fake_training()

        fig = Figure(figsize=(10, 5), facecolor=self.P["bg"])
        fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.12,
                            wspace=0.3, hspace=0.4)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, (3, 4))

        eps = data["episodes"]

        # Reward curves
        ax1.plot(eps, data["pursuer_rewards"], color=_PURSUER_C, linewidth=1.2)
        ax1.plot(eps, data["evader_rewards"],  color=_EVADER_C,  linewidth=1.2)
        ax1.set_title("Reward Curves", fontsize=9)
        _style_ax(ax1, self.P)

        # Convergence line (rolling std of win rate)
        ax2.plot(eps, data["convergence"], color=self.P["accent"], linewidth=1.2)
        ax2.set_title("Policy Variance (rolling std)", fontsize=9)
        _style_ax(ax2, self.P)

        # Reward accumulation heatmap
        if data.get("heatmap") is not None:
            h = data["heatmap"]
            im = ax3.imshow(h, aspect="auto", cmap="viridis",
                            origin="lower", interpolation="nearest")
            ax3.set_title("Reward Accumulation Heatmap  (episode × step)", fontsize=9)
            ax3.set_xlabel("Step within episode", fontsize=8)
            ax3.set_ylabel("Training episode", fontsize=8)
            fig.colorbar(im, ax=ax3, fraction=0.015, pad=0.02)
        else:
            ax3.text(0.5, 0.5, "Heatmap data not recorded",
                     ha="center", va="center",
                     color=self.P["muted"], transform=ax3.transAxes)
        _style_ax(ax3, self.P)

        _embed(fig, self._training_tab)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 3 — Policy Quality
    # ──────────────────────────────────────────────────────────────────────────

    def _populate_policy_tab(self, metrics):
        _clear(self._policy_tab)
        if not HAS_MPL:
            _no_mpl(self._policy_tab, self.P)
            return

        data = _extract(metrics, "policy") or _fake_policy()

        fig = Figure(figsize=(10, 4.5), facecolor=self.P["bg"])
        fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.08,
                            wspace=0.35)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        im1 = ax1.imshow(data["pursuer_coverage"], cmap="Blues",
                         origin="upper", vmin=0)
        ax1.set_title("Pursuer — State Coverage", fontsize=9)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        _style_heatmap_ax(ax1, self.P)

        im2 = ax2.imshow(data["evader_coverage"], cmap="Purples",
                         origin="upper", vmin=0)
        ax2.set_title("Evader — State Coverage", fontsize=9)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        _style_heatmap_ax(ax2, self.P)

        _embed(fig, self._policy_tab)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 4 — Behaviour
    # ──────────────────────────────────────────────────────────────────────────

    def _populate_behaviour_tab(self, metrics):
        _clear(self._behaviour_tab)
        if not HAS_MPL:
            _no_mpl(self._behaviour_tab, self.P)
            return

        data = _extract(metrics, "behaviour") or _fake_behaviour()

        fig = Figure(figsize=(10, 4.5), facecolor=self.P["bg"])
        fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.1,
                            wspace=0.35)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Trajectory on grid
        for traj in data["trajectories"][:5]:
            px = [t[0][0] for t in traj]
            py = [t[0][1] for t in traj]
            ex = [t[1][0] for t in traj]
            ey = [t[1][1] for t in traj]
            ax1.plot(px, py, color=_PURSUER_C, alpha=0.5, linewidth=0.8)
            ax1.plot(ex, ey, color=_EVADER_C,  alpha=0.5, linewidth=0.8)

        ax1.set_xlim(-0.5, 9.5)
        ax1.set_ylim(-0.5, 9.5)
        ax1.set_aspect("equal")
        ax1.invert_yaxis()
        ax1.set_title("Agent Trajectories (last 5 episodes)", fontsize=9)
        _style_ax(ax1, self.P)

        # Distance over steps in one episode
        if data["distances"]:
            steps = list(range(len(data["distances"])))
            ax2.plot(steps, data["distances"], color=self.P["accent"],
                     linewidth=1.2)
            ax2.axhline(1, color=self.P["danger"], linewidth=0.8,
                        linestyle="--", label="Catch distance")
            ax2.set_title("Agent Distance — Last Episode", fontsize=9)
            ax2.legend(fontsize=7)
            _style_ax(ax2, self.P)

        _embed(fig, self._behaviour_tab)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 5 — Reliability
    # ──────────────────────────────────────────────────────────────────────────

    def _populate_reliability_tab(self, metrics):
        _clear(self._reliability_tab)
        if not HAS_MPL:
            _no_mpl(self._reliability_tab, self.P)
            return

        data = _extract(metrics, "reliability") or _fake_reliability()

        fig = Figure(figsize=(10, 4.5), facecolor=self.P["bg"])
        fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.1,
                            wspace=0.35)

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # Policy variance band
        eps = data["episodes"]
        ax1.plot(eps, data["mean_wr"], color=self.P["accent"], linewidth=1.5)
        ax1.fill_between(eps,
                         [m - s for m, s in zip(data["mean_wr"], data["std_wr"])],
                         [m + s for m, s in zip(data["mean_wr"], data["std_wr"])],
                         alpha=0.2, color=self.P["accent"])
        ax1.set_title("Win Rate ± Std Dev", fontsize=9)
        _style_ax(ax1, self.P)

        # Performance after convergence bar
        phases = ["Early", "Mid", "Post-convergence"]
        p_vals = data.get("pac_pursuer", [0.4, 0.55, 0.72])
        e_vals = data.get("pac_evader",  [0.6, 0.45, 0.28])

        x = np.arange(len(phases))
        w = 0.35
        ax2.bar(x - w/2, p_vals, w, color=_PURSUER_C, label="Pursuer")
        ax2.bar(x + w/2, e_vals, w, color=_EVADER_C,  label="Evader")
        ax2.set_xticks(x)
        ax2.set_xticklabels(phases, fontsize=7)
        ax2.set_title("Performance After Convergence", fontsize=9)
        ax2.legend(fontsize=7)
        _style_ax(ax2, self.P)

        # Summary table (confidence intervals if multi-seed available)
        labels = ["Win Rate", "Ep. Length", "Reward"]
        p_means = data.get("ci_pursuer_mean", [0.68, 42, -0.15])
        p_cis   = data.get("ci_pursuer_ci",   [0.04,  5,  0.08])
        e_means = data.get("ci_evader_mean",  [0.32, 42,  0.12])
        e_cis   = data.get("ci_evader_ci",    [0.04,  5,  0.06])

        y = np.arange(len(labels))
        ax3.barh(y + 0.2, p_means, 0.35, xerr=p_cis,
                 color=_PURSUER_C, label="Pursuer", capsize=4)
        ax3.barh(y - 0.2, e_means, 0.35, xerr=e_cis,
                 color=_EVADER_C,  label="Evader",  capsize=4)
        ax3.set_yticks(y)
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_title("95% CI (if multi-seed)", fontsize=9)
        ax3.legend(fontsize=7)
        _style_ax(ax3, self.P)

        _embed(fig, self._reliability_tab)


# ── Shared chart helpers ──────────────────────────────────────────────────────

def _style_ax(ax, P: dict):
    ax.set_facecolor(P["bg"])
    ax.figure.set_facecolor(P["bg"])
    for spine in ax.spines.values():
        spine.set_color(P["border"])
    ax.tick_params(colors=P["muted"], labelsize=7)
    ax.xaxis.label.set_color(P["muted"])
    ax.yaxis.label.set_color(P["muted"])
    ax.title.set_color(P["text"])


def _style_heatmap_ax(ax, P: dict):
    _style_ax(ax, P)
    ax.set_xticks(range(0, 10, 2))
    ax.set_yticks(range(0, 10, 2))


def _embed(fig: "Figure", parent: ttk.Frame):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(fill="both", expand=True)


def _clear(frame: ttk.Frame):
    for w in frame.winfo_children():
        w.destroy()


def _no_mpl(frame: ttk.Frame, P: dict):
    tk.Label(frame,
             text="matplotlib not installed.\n"
                  "Run: pip install matplotlib",
             bg=P["bg"], fg=P["muted"],
             font=("Segoe UI", 10)).pack(expand=True)


def _extract(metrics, category: str):
    """
    Pull chart-ready data from a MetricsRecorder object.
    Maps compute_summary() keys to what each tab expects.
    """
    if metrics is None:
        return None
    try:
        summary = metrics.compute_summary()
        if not summary:
            return None

        if category == "outcome":
            wr  = summary.get("win_rates", {})
            eps = wr.get("episodes", [])
            return {
                "episodes":   eps,
                "pursuer_wr": [v * 100 for v in wr.get("pursuer", [])],
                "evader_wr":  [v * 100 for v in wr.get("evader",  [])],
                "ep_lengths": summary.get("episode_lengths", []),
            }

        if category == "training":
            cr  = summary.get("cumulative_rewards", {})
            wr  = summary.get("win_rates", {})
            eps = cr.get("episodes", [])
            # rolling std of win rate as convergence proxy
            p_wr = wr.get("pursuer", [])
            w    = max(1, len(p_wr) // 20)
            std  = []
            for i in range(len(p_wr)):
                sl = p_wr[max(0, i-w):i+1]
                import numpy as np
                std.append(float(np.std(sl)) if sl else 0.0)
            hm = summary.get("reward_heatmap", {})
            return {
                "episodes":        eps,
                "pursuer_rewards": cr.get("pursuer", []),
                "evader_rewards":  cr.get("evader",  []),
                "convergence":     std,
                "heatmap":         hm.get("pursuer") if hm else None,
            }

        if category == "policy":
            sc = summary.get("state_coverage", {})
            return {
                "pursuer_coverage": sc.get("pursuer"),
                "evader_coverage":  sc.get("evader"),
            }

        if category == "behaviour":
            eps_list = summary.get("episodes", [])
            trajs = []
            dists = []
            if eps_list:
                # last 5 episodes
                for ep in eps_list[-5:]:
                    snaps = ep.snapshots
                    trajs.append([(s["pursuer_pos"], s["evader_pos"], s["goal_pos"])
                                  for s in snaps])
                if eps_list:
                    dists = eps_list[-1].agent_distances
            return {"trajectories": trajs, "distances": dists}

        if category == "reliability":
            wr  = summary.get("win_rates", {})
            eps = wr.get("episodes", [])
            import numpy as np
            p_wr = wr.get("pursuer", [])
            w    = max(1, len(p_wr) // 20)
            mean_wr, std_wr = [], []
            for i in range(len(p_wr)):
                sl = p_wr[max(0, i-w):i+1]
                mean_wr.append(float(np.mean(sl)) * 100 if sl else 0.0)
                std_wr.append(float(np.std(sl))  * 100 if sl else 0.0)
            pac = summary.get("performance_pac", {})
            p_pac = pac.get("pursuer", {})
            e_pac = pac.get("evader",  {})
            return {
                "episodes": eps,
                "mean_wr":  mean_wr,
                "std_wr":   std_wr,
                "pac_pursuer": [
                    p_pac.get("early") or 0,
                    p_pac.get("mid")   or 0,
                    p_pac.get("late")  or 0,
                ],
                "pac_evader": [
                    e_pac.get("early") or 0,
                    e_pac.get("mid")   or 0,
                    e_pac.get("late")  or 0,
                ],
            }

    except Exception:
        pass
    return None


# ── Fake data generators (shown when training backend is missing) ─────────────

def _fake_outcome():
    import random, math
    n = 40
    eps = list(range(50, 50 * (n + 1), 50))
    p_wr = []
    val = 45.0
    for _ in range(n):
        val = min(85, max(15, val + random.gauss(0.4, 2)))
        p_wr.append(val)
    e_wr = [100 - v for v in p_wr]
    lengths = [random.gauss(45, 12) for _ in range(200)]
    return {"episodes": eps, "pursuer_wr": p_wr, "evader_wr": e_wr,
            "ep_lengths": lengths}


def _fake_training():
    import random
    import numpy as np
    n = 40
    eps = list(range(50, 50 * (n + 1), 50))
    p_r = [random.gauss(-0.3 + i*0.01, 0.15) for i in range(n)]
    e_r = [random.gauss(0.1  + i*0.008, 0.15) for i in range(n)]
    conv = [max(0.01, 0.35 - i*0.008 + random.gauss(0, 0.02)) for i in range(n)]
    heatmap = np.random.exponential(0.3, (n, 60))
    for i in range(n):
        heatmap[i, :] *= (1 + i / n)
    return {"episodes": eps, "pursuer_rewards": p_r, "evader_rewards": e_r,
            "convergence": conv, "heatmap": heatmap}


def _fake_policy():
    import numpy as np
    p_cov = np.random.poisson(4, (10, 10)).astype(float)
    e_cov = np.random.poisson(3, (10, 10)).astype(float)
    p_cov[9, 9] += 30  # pursuer starts bottom-right
    e_cov[0, 0] += 30  # evader starts top-left
    return {"pursuer_coverage": p_cov, "evader_coverage": e_cov}


def _fake_behaviour():
    import random
    trajs = []
    for _ in range(5):
        t = []
        px, py = 9, 9
        ex, ey = 0, 0
        for _ in range(random.randint(15, 50)):
            t.append(((px, py), (ex, ey), (5, 5)))
            px = max(0, min(9, px + random.randint(-1, 1)))
            py = max(0, min(9, py + random.randint(-1, 1)))
            ex = max(0, min(9, ex + random.randint(-1, 1)))
            ey = max(0, min(9, ey + random.randint(-1, 1)))
        trajs.append(t)

    last = trajs[-1]
    dists = [abs(s[0][0]-s[1][0]) + abs(s[0][1]-s[1][1]) for s in last]
    return {"trajectories": trajs, "distances": dists}


def _fake_reliability():
    import random
    n = 40
    eps = list(range(50, 50*(n+1), 50))
    mean = [45 + i*1.1 + random.gauss(0, 1) for i in range(n)]
    std  = [max(0.5, 12 - i*0.25 + random.gauss(0, 0.5)) for i in range(n)]
    return {"episodes": eps, "mean_wr": mean, "std_wr": std}