"""Visualization helper:

This module implements a plotting class that updates as trials are produced by a
generator. Pass a trial_iter (an iterator yielding (m, b, r, epoch_idx)) to the
constructor to animate discovery live. If trial_iter is None, the class can also
accept precomputed trials via the attribute `trials` (not required here).
"""
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class RegressionPlot:
    def __init__(self, x_all, y_all, x_train, y_train, x_test, y_test,
                 epochs, test_rss=None, rss_a=None, rss_b=None, cur_rss=None, trial_iter=None,
                 external_producer: bool = False, title: str = None, x_label: str = None, y_label: str = None):
        self.x_all = x_all
        self.y_all = y_all
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.test_rss = test_rss
        # optional final metrics (set by an external producer when available)
        self.train_rmse = None
        self.test_rmse = None
        self.train_r2 = None
        self.test_r2 = None
        self.rss_a = rss_a
        self.rss_b = rss_b
        self.cur_rss = cur_rss

        # streaming iterator that yields (m, b, r, epoch_idx)
        self.trial_iter = iter(trial_iter) if trial_iter is not None else None
        # streaming True if we have an iterator OR if an external producer will
        # push trials via add_trial(). Use external_producer=True to enable
        # this mode.
        self.streaming = (trial_iter is not None) or bool(external_producer)

        # place to accumulate trials by epoch when streaming
        self.trials_by_epoch = []
        # pending trials pushed by external producers (thread-safe append in CPython)
        self._pending = []

        # figure + axes
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.scatter(self.x_train, self.y_train, alpha=0.7, label='train')
        self.ax.scatter(self.x_test, self.y_test, alpha=0.7, label='test')
        # axis labels and title may be dataset-specific; accept overrides
        if x_label is None:
            x_label = 'x'
        if y_label is None:
            y_label = 'y'
        if title is None:
            title = 'y vs x'
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)

        # artists for lines
        self.x0, self.x1 = float(self.x_all.min()) - 1.0, float(self.x_all.max()) + 1.0
        self.ln_acc, = self.ax.plot([], [], 'r-', linewidth=2, label='best-so-far')
        self.ln_trial, = self.ax.plot([], [], 'b--', linewidth=1, alpha=0.9, label='trial')
        self.ln_best, = self.ax.plot([], [], 'g-', linewidth=1.5, alpha=0.9, label='best-overall')

        # legend and info text box
        self.legend = self.ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), bbox_transform=self.ax.transAxes)
        self.text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, ha='left', va='top',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        self.ax.set_xlim(self.x0, self.x1)
        self.ax.set_ylim(self.y_all.min() - 1000, self.y_all.max() + 1000)

        # animation state
        self.state = {'idx': 0, 'best': (None, None, float('inf')), 'hold': 0, 'sampled': [], 'epoch_shown': -1}
        self.hold_frames = 40

        # inset axes showing RSS vs slope
        self.inset_ax = self.fig.add_axes([0.6, 0.07, 0.35, 0.35])
        self.inset_ax.set_title('RSS vs slope (current pass)')
        self.num_epochs = len(self.epochs)
        self.cmap = plt.get_cmap('tab10', max(1, self.num_epochs))
        self.inset_scatter = self.inset_ax.scatter([], [], c=[], cmap=self.cmap, s=30, edgecolors='face', vmin=0, vmax=max(0, self.num_epochs-1))
        self.inset_marker, = self.inset_ax.plot([], [], 'ro')
        self.inset_ax.set_xlabel('slope')
        self.inset_ax.set_ylabel('RSS')

        # initial inset limits (conservative); streaming updates them per-epoch
        self.inset_ax.set_xlim(-100, 400)
        self.inset_ax.set_ylim(0, max(1.0, float(np.median(self.y_all)**2)))

        # FuncAnimation handle
        self.ani = None
        # external-producer completion flag
        self._producer_done = False
        # whether the figure is currently being displayed (show() in progress)
        self._displaying = False


    def _init(self):
        self.ln_acc.set_data([], [])
        self.ln_trial.set_data([], [])
        self.ln_best.set_data([], [])
        # position info box to the right of legend when possible
        try:
            renderer = self.fig.canvas.get_renderer()
            leg_bbox = self.legend.get_window_extent(renderer)
            inv = self.ax.transAxes.inverted()
            leg_bbox_axes = inv.transform_bbox(leg_bbox)
            pad = 0.01
            new_x = leg_bbox_axes.x1 + pad
            new_y = leg_bbox_axes.y1
            new_x = min(new_x, 0.98)
            new_y = min(new_y, 0.98)
            self.text.set_position((new_x, new_y))
            self.text.set_ha('left')
            self.text.set_va('top')
        except Exception:
            pass
        self.text.set_text('')
        self.inset_scatter.set_offsets(np.empty((0, 2)))
        self.inset_scatter.set_array(np.array([]))
        self.inset_marker.set_data([], [])
        return (self.ln_acc, self.ln_trial, self.ln_best, self.text, self.inset_scatter, self.inset_marker)

    def _process_trial(self, m, b, r, epoch_idx):
        """Shared trial processing extracted from _update so external producers can call it.

        This only mutates simple Python structures (lists, dicts) and matplotlib
        artists; it is intended to be called from the main thread (animation
        callback). External producer threads should use `add_trial` which only
        appends to a thread-safe list; the animation will pick those up.
        """
        # ensure storage exists for this epoch
        while len(self.trials_by_epoch) <= epoch_idx:
            self.trials_by_epoch.append([])
        self.trials_by_epoch[epoch_idx].append((m, b, r))

        # Plot the trial line
        y0_try = m * self.x0 + b
        y1_try = m * self.x1 + b
        self.ln_trial.set_data([self.x0, self.x1], [y0_try, y1_try])

        # update best-so-far
        if r < self.state['best'][2]:
            self.state['best'] = (m, b, r)
            y0_best = m * self.x0 + b
            y1_best = m * self.x1 + b
            self.ln_best.set_data([self.x0, self.x1], [y0_best, y1_best])

        if self.state['best'][0] is not None:
            mb, bb, rb = self.state['best']
            y0_acc = mb * self.x0 + bb
            y1_acc = mb * self.x1 + bb
            self.ln_acc.set_data([self.x0, self.x1], [y0_acc, y1_acc])

        # update info text: prefer explicit step stored in epoch config, else compute from slope_values
        epoch_conf = self.epochs[epoch_idx]
        step_size = epoch_conf.get('step', None)
        if not step_size:
            slope_vals = epoch_conf.get('slope_values', None)
            if slope_vals is None:
                # backward-compat fallback
                slope_vals = epoch_conf.get('ms', None)
            if slope_vals is not None and len(slope_vals) > 1:
                step_size = float(slope_vals[1] - slope_vals[0])
            else:
                step_size = 0.0
        self.text.set_text(
            f'epoch={epoch_idx} trial={self.state["idx"]}\n'
            f'slope={m:.2f}\n'
            f'trial_rss={r:.1f}\n'
            f'best_rss={self.state["best"][2]:.1f}\n'
            f'step ≈ {step_size:.3f}'
        )

        # append sampled point and update inset scatter
        self.state['sampled'].append((m, r, epoch_idx))
        sampled_slopes = [s for s, rr, e in self.state['sampled']]
        sampled_rss = [rr for s, rr, e in self.state['sampled']]
        if len(sampled_slopes) > 0:
            pts = np.column_stack((sampled_slopes, sampled_rss))
            self.inset_scatter.set_offsets(pts)
            colors = np.array([e for s, rr, e in self.state['sampled']], dtype=float)
            self.inset_scatter.set_array(colors)
            self.inset_scatter.set_clim(0, max(0, self.num_epochs-1))
        else:
            self.inset_scatter.set_offsets(np.empty((0, 2)))
            self.inset_scatter.set_array(np.array([]))
        self.inset_marker.set_data([m], [r])

        # keep initial epoch points visible (epoch 0 special-case)
        if epoch_idx == 0:
            epoch0 = [(s, rr) for s, rr, e in self.state['sampled'] if e == 0]
            if epoch0:
                s_vals = np.array([s for s, rr in epoch0])
                y_vals = np.array([rr for s, rr in epoch0])
                xmin, xmax = self.inset_ax.get_xlim()
                ymin, ymax = self.inset_ax.get_ylim()
                s_min, s_max = float(s_vals.min()), float(s_vals.max())
                y_min, y_max = float(y_vals.min()), float(y_vals.max())
                if (s_min < xmin) or (s_max > xmax) or (y_min < ymin) or (y_max > ymax):
                    xr = s_max - s_min
                    pad_x = xr * 0.08 if xr != 0 else 1.0
                    new_xmin = s_min - pad_x
                    new_xmax = s_max + pad_x
                    self.inset_ax.set_xlim(new_xmin, new_xmax)
                    yr = y_max - y_min
                    pad_y = yr * 0.12 if yr != 0 else max(1000.0, y_max * 0.05)
                    self.inset_ax.set_ylim(y_min - pad_y, y_max + pad_y)

        # epoch zoom handled elsewhere when epoch changes


    def _update(self, frame):
        # First, consume any pending trials pushed by an external producer
        processed = 0
        if self._pending:
            # drain pending into a local list and process in arrival order
            pending = list(self._pending)
            self._pending.clear()
            for m, b, r, epoch_idx in pending:
                self._process_trial(m, b, r, epoch_idx)
                self.state['idx'] += 1
                processed += 1

        # Pull next trial from streaming iterator or use precomputed list
        try:
            if self.streaming and self.trial_iter is not None:
                nxt = next(self.trial_iter)
                m, b, r, epoch_idx = nxt
                # process via shared helper
                self._process_trial(m, b, r, epoch_idx)
                self.state['idx'] += 1
            elif self.streaming and self.trial_iter is None:
                # external-producer mode: if nothing was processed this frame,
                # either wait for more trials or, if producer finished, stop.
                if processed == 0:
                    if self._producer_done:
                        raise StopIteration
                    # no new data this frame -> nothing to draw
                    return (self.ln_acc, self.ln_trial, self.ln_best, self.text, self.inset_scatter, self.inset_marker)
            else:
                # not streaming: assume attribute `trials` exists
                if self.state['idx'] >= len(self.trials):
                    raise StopIteration
                m, b, r, epoch_idx = self.trials[self.state['idx']]
                self._process_trial(m, b, r, epoch_idx)
                self.state['idx'] += 1
        except StopIteration:
            # generator finished -> show final best for a few frames then stop
            if self.state['hold'] < self.hold_frames:
                if self.state['best'][0] is not None:
                    mb, bb, rb = self.state['best']
                    y0_acc = mb * self.x0 + bb
                    y1_acc = mb * self.x1 + bb
                    self.ln_acc.set_data([self.x0, self.x1], [y0_acc, y1_acc])
                    self.ln_trial.set_data([], [])
                    self.ln_best.set_data([], [])
                    test_str = f"{self.test_rss:.1f}" if (self.test_rss is not None) else "N/A"
                    train_rmse_str = f"{self.train_rmse:.1f}" if (self.train_rmse is not None) else "N/A"
                    test_rmse_str = f"{self.test_rmse:.1f}" if (self.test_rmse is not None) else "N/A"
                    train_r2_str = f"{self.train_r2:.4f}" if (self.train_r2 is not None) else "N/A"
                    test_r2_str = f"{self.test_r2:.4f}" if (self.test_r2 is not None) else "N/A"
                    self.text.set_text(
                        f'final\nslope={mb:.2f}\nintercept={bb:.1f}\n'
                        f'train_rss={rb:.1f}\ntrain_rmse={train_rmse_str}\ntrain_r2={train_r2_str}\n'
                        f'test_rss={test_str}\ntest_rmse={test_rmse_str}\ntest_r2={test_r2_str}'
                    )
                self.state['hold'] += 1
                return (self.ln_acc, self.ln_trial, self.ln_best, self.text, self.inset_scatter, self.inset_marker)
            try:
                self.ani.event_source.stop()
            except Exception:
                pass
            return (self.ln_acc, self.ln_trial, self.ln_best, self.text, self.inset_scatter, self.inset_marker)

        # if we fell through to here (streaming iterator produced a trial),
        # the processing was already done inside _process_trial above and
        # state['idx'] was incremented there.

        # epoch zoom: when a new epoch starts, prefer 'slope_values' in epoch-config, fallback to 'ms'
        if self.state.get('epoch_shown') != epoch_idx:
            self.state['epoch_shown'] = epoch_idx
            epoch_config = self.epochs[epoch_idx]
            # prefer 'slope_values' (new name), fallback to 'ms' for older configs
            epoch_slopes = None
            if 'slope_values' in epoch_config and epoch_config['slope_values'] is not None:
                epoch_slopes = np.array(epoch_config['slope_values'])
            elif 'ms' in epoch_config and epoch_config['ms'] is not None:
                epoch_slopes = np.array(epoch_config['ms'])
            if epoch_slopes is not None:
                epoch_rss_vals = np.array([t[2] for t in self.trials_by_epoch[epoch_idx]]) if len(self.trials_by_epoch) > epoch_idx and len(self.trials_by_epoch[epoch_idx])>0 else None
                xr = epoch_slopes.max() - epoch_slopes.min()
                pad_x = xr * 0.08 if xr != 0 else 1.0
                new_xmin = epoch_slopes.min() - pad_x
                new_xmax = epoch_slopes.max() + pad_x
                self.inset_ax.set_xlim(new_xmin, new_xmax)
                in_window = [(s, rr) for s, rr, e in self.state['sampled'] if new_xmin <= s <= new_xmax]
                if in_window:
                    ys = np.array([rr for s, rr in in_window])
                    yr = ys.max() - ys.min()
                    pad_y = yr * 0.12 if yr != 0 else max(1000.0, ys.max() * 0.05)
                    self.inset_ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
                elif epoch_rss_vals is not None and len(epoch_rss_vals)>0:
                    yr = epoch_rss_vals.max() - epoch_rss_vals.min()
                    pad_y = yr * 0.12 if yr != 0 else max(1000.0, epoch_rss_vals.max() * 0.05)
                    self.inset_ax.set_ylim(epoch_rss_vals.min() - pad_y, epoch_rss_vals.max() + pad_y)
            else:
                # fallback: use collected trials for this epoch if available
                if len(self.trials_by_epoch) > epoch_idx and len(self.trials_by_epoch[epoch_idx])>0:
                    epoch_trials = self.trials_by_epoch[epoch_idx]
                    epoch_slopes = np.array([t[0] for t in epoch_trials])
                    epoch_rss_vals = np.array([t[2] for t in epoch_trials])
                    xr = epoch_slopes.max() - epoch_slopes.min()
                    pad_x = xr * 0.08 if xr != 0 else 1.0
                    new_xmin = epoch_slopes.min() - pad_x
                    new_xmax = epoch_slopes.max() + pad_x
                    self.inset_ax.set_xlim(new_xmin, new_xmax)
                    yr = epoch_rss_vals.max() - epoch_rss_vals.min()
                    pad_y = yr * 0.12 if yr != 0 else 1000.0
                    self.inset_ax.set_ylim(epoch_rss_vals.min() - pad_y, epoch_rss_vals.max() + pad_y)

        self.state['idx'] += 1
        return (self.ln_acc, self.ln_trial, self.ln_best, self.text, self.inset_scatter, self.inset_marker)

    # Public API for external producers -------------------------------------------------
    def add_trial(self, m, b, r, epoch_idx):
        """Quickly enqueue a trial produced by an external thread or process.

        This method is intended to be called from a background thread; it only
        appends to an in-memory list which the animation loop drains.
        """
        # debug: indicate a trial was enqueued
        # print(f'add_trial enqueued: m={m:.3f} epoch={epoch_idx} r={r:.1f}')
        self._pending.append((m, b, r, epoch_idx))

    def is_displaying(self):
        """Return True when the plot's show() has been invoked and the figure
        is being displayed to the user. This allows producers to pace only when
        users can see updates.
        """
        return bool(self._displaying)

    def producer_done(self):
        """Signal that the external producer has finished producing trials."""
        self._producer_done = True

    

    def run(self):
        # If streaming, use an indefinite frame counter and the generator will stop the animation
        if self.streaming:
            frames = itertools.count()
        else:
            frames = range(len(self.trials) + self.hold_frames + 5)
        # cache_frame_data must be disabled for an unbounded/streaming frames generator
        # to avoid unbounded memory usage and the Matplotlib UserWarning.
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=frames,
            init_func=self._init,
            blit=True,
            interval=80,
            repeat=False,
            cache_frame_data=False,
        )
        try:
            # mark that we're about to display the figure so external producers
            # can choose to apply pacing/delays only when the GUI is visible.
            self._displaying = True
            try:
                plt.show()
            finally:
                # show() returned (window closed) -> no longer displaying
                self._displaying = False
        except Exception:
            # compute a reasonable root for saving
            ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            out_gif = os.path.join(ROOT, 'manual_regression_sequence.gif')
            try:
                self.ani.save(out_gif, writer='pillow')
                print('No display available; saved animation to', out_gif)
            except Exception as e2:
                print('Could not display or save animation:', e2)
