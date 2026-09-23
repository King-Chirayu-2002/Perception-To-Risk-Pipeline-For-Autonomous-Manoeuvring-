import numpy as np
import matplotlib.pyplot as plt


class EvaluationMetrics:

    def __init__(self):

        self.track_history = {}
        self.id_switches = 0
        self.prev_ids = set()
        self.ttc_values = []
        self.velocities = []


    # =========================
    # Update tracking history
    # =========================
    def update_tracks(self, tracks):

        current_ids = set()

        for trk in tracks:

            x1, y1, x2, y2, tid = trk.astype(int)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            current_ids.add(tid)

            if tid not in self.track_history:
                self.track_history[tid] = []

            self.track_history[tid].append((cx, cy))

            # velocity estimation
            if len(self.track_history[tid]) > 1:

                p1 = self.track_history[tid][-2]
                p2 = self.track_history[tid][-1]

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                vel = np.sqrt(dx**2 + dy**2)

                self.velocities.append(vel)

        # ID switch detection
        lost_ids = self.prev_ids - current_ids

        if lost_ids:
            self.id_switches += len(lost_ids)

        self.prev_ids = current_ids


    # =========================
    # Store TTC values
    # =========================
    def add_ttc(self, ttc):

        if ttc > 0 and ttc < 100:
            self.ttc_values.append(ttc)


    # =========================
    # Trajectory smoothness
    # =========================
    def trajectory_smoothness(self):

        scores = []

        for tid in self.track_history:

            history = self.track_history[tid]

            if len(history) < 3:
                continue

            diffs = []

            for i in range(1, len(history)):

                dx = history[i][0] - history[i-1][0]
                dy = history[i][1] - history[i-1][1]

                diffs.append(np.sqrt(dx**2 + dy**2))

            scores.append(np.var(diffs))

        if len(scores) == 0:
            return 0

        return np.mean(scores)


    # =========================
    # Print evaluation results
    # =========================
    def print_results(self):

        print("\n===== Evaluation Results =====")

        print("Total objects tracked:", len(self.track_history))

        print("ID switches:", self.id_switches)

        if len(self.velocities) > 0:
            print("Average velocity:", np.mean(self.velocities))

        if len(self.ttc_values) > 0:

            print("Average TTC:", np.mean(self.ttc_values))
            print("Minimum TTC:", np.min(self.ttc_values))

        print("Trajectory smoothness:", self.trajectory_smoothness())


    # =========================
    # TTC risk graph
    # =========================
    def plot_ttc_graph(self, save_path="ttc_risk_graph.png"):

        if len(self.ttc_values) == 0:
            print("No TTC data to plot.")
            return

        frames = np.arange(len(self.ttc_values))
        ttc = np.array(self.ttc_values)

        plt.figure(figsize=(9,4))

        # Safe zone
        safe = ttc > 4
        plt.scatter(frames[safe], ttc[safe], color='green', label="Safe TTC")

        # Medium risk
        medium = (ttc > 2) & (ttc <= 4)
        plt.scatter(frames[medium], ttc[medium], color='orange', label="Moderate Risk")

        # High risk
        danger = ttc <= 2
        plt.scatter(frames[danger], ttc[danger], color='red', label="High Collision Risk")

        # Plot line
        plt.plot(frames, ttc, color='black', alpha=0.4)

        # Danger threshold
        plt.axhline(2, linestyle="--", color="red", label="Danger Threshold (2s)")

        plt.xlabel("Frame Index")
        plt.ylabel("Time-To-Collision (seconds)")
        plt.title("Collision Risk Over Time (TTC)")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(save_path)

        print("TTC risk graph saved as:", save_path)

        plt.close()


    # =========================
    # Global trajectory map
    # =========================
    def plot_global_trajectories(self, save_path="global_trajectory_map.png"):

        if len(self.track_history) == 0:
            print("No trajectories to plot.")
            return

        plt.figure(figsize=(7,7))

        colors = plt.cm.tab20(
            np.linspace(0,1,len(self.track_history))
        )

        for i, tid in enumerate(self.track_history):

            history = self.track_history[tid]

            if len(history) < 2:
                continue

            xs = [p[0] for p in history]
            ys = [p[1] for p in history]

            color = colors[i]

            # trajectory line
            plt.plot(xs, ys, linewidth=2, color=color)

            # start marker
            plt.scatter(xs[0], ys[0], color='green', s=40)

            # end marker
            plt.scatter(xs[-1], ys[-1], color='red', s=40)

        plt.gca().invert_yaxis()

        plt.xlabel("Image X")
        plt.ylabel("Image Y")

        plt.title("Global Object Trajectories")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(save_path)

        print("Trajectory map saved as:", save_path)

        plt.close()