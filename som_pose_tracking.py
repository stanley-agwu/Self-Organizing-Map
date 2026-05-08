# ================================================================
# Competitive Learning (SOM-like) model training and testing
# Evaluates discriminant scores across 3 trained SOM models (10,12,14 clusters)
# Filters short non-consecutive postures (<10 samples)
# visualizes cluster discriminant trends
# ================================================================

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from matplotlib.lines import Line2D
from data_preprocessing import load_training_data, load_test_data


def compute_grid_shape(num_neurons: int) -> tuple[int, int]:
    rows = int(np.sqrt(num_neurons))
    cols = int(np.ceil(num_neurons / rows))
    return rows, cols


# ================================================================
# Competitive Learning Class
# ================================================================
class CompetitiveLearning:
    def __init__(
        self, num_neurons, training_data, radius, learning_rate, gaussian, dist_metric
    ):
        self.num_neurons = num_neurons
        self.radius = radius
        self.learning_rate = learning_rate
        self.gaussian = gaussian
        self.dist_metric = dist_metric
        self.grid_shape = compute_grid_shape(num_neurons)

        self.input_data = load_training_data(training_data)
        self.neuron_weights = np.random.normal(
            np.mean(self.input_data),
            np.std(self.input_data),
            size=(self.num_neurons, len(self.input_data[0])),
        )

        # Containers
        self.potential = np.ones(self.num_neurons)
        self.activation = np.ones(self.num_neurons)
        self.winners_list = []
        self.error = []
        self.topographic_error = []
        self.epoch_weights = []
        self.pca_results = []
        self.average_distance = []
        self.convergence_counter = 0

        # Visualizations
        self.joint_connections = [
            (0, 1),
            (1, 2),
            (2, 3),  # right arm
            (0, 4),
            (4, 5),
            (5, 6),  # left arm
            (0, 7),
            (7, 8),
            (8, 9),  # spine
            (7, 10),
            (10, 11),  # right leg
            (7, 12),
            (12, 13),  # left leg
        ]

    # ------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------
    def calculate_distances(self, input_vector, neuron_weights):
        """
            Compute the distance between an input posture sample and all neuron
            prototype weight vectors using the selected distance metric.

            This function evaluates how similar or dissimilar the input vector is
            to each neuron in the Self-Organizing Map (SOM). The resulting distance
            values are used during Best Matching Unit (BMU) selection, clustering,
            posture classification, and confidence/discriminant calculations.

            Supported distance metrics:
                - "manhattan" : Manhattan (L1) distance
                - "minkowski" : Minkowski distance with p=3
                - "hamming"   : Hamming distance
                - "cosine"    : Cosine distance
                - "euclidean" : Euclidean (L2) distance

            If an unsupported metric is specified, cosine distance is used by default.

            Args:
                input_vector (np.ndarray):
                    Input posture feature vector representing one sample.

                neuron_weights (np.ndarray):
                    Array of neuron prototype vectors with shape:
                        (num_neurons, num_features)

            Returns:
                list[float]:
                    Distance from the input vector to each neuron prototype.
                    The returned list has length equal to the number of neurons,
                    where smaller values indicate higher similarity.
            """

        distance_functions = {
            "manhattan": distance.cityblock,
            "minkowski": lambda x, y: distance.minkowski(x, y, p=3),
            "hamming": distance.hamming,
            "cosine": distance.cosine,
            "euclidean": distance.euclidean,
        }

        # Use chosen metric, default to Euclidean
        distance_function = distance_functions.get(self.dist_metric, distance.cosine)

        # Compute distance from input vector to each neuron prototype
        return [
            distance_function(weight_vector, input_vector)
            for weight_vector in neuron_weights
        ]

    def find_winner_neuron(self):  # Returns winning neuron index and distance weight
        """Find the neuron with minimum distance (BMU)."""
        avg_dist = np.average(self.distance)
        self.average_distance.append(avg_dist)
        neuron_idx = np.argmin(self.distance)
        return neuron_idx, np.min(self.distance)

    def find_two_bmus(self, input_vector):
        """
        Return:
            bmu_index       -> Best Matching Unit
            second_bmu_index -> 2nd closest neuron
            bmu_distance -> bmu neuron distance
            second_bmu_distance -> 2nd bmu neuron distance
            distances       -> all neuron distances
        """
        distances = self.calculate_distances(input_vector, self.neuron_weights)

        sorted_neuron_indices = np.argsort(distances)

        bmu = sorted_neuron_indices[0]
        second_bmu = sorted_neuron_indices[1]

        bmu_distance = distances[bmu]
        second_bmu_distance = distances[second_bmu]

        return (bmu, second_bmu, bmu_distance, second_bmu_distance, distances)

    # ------------------------------------------------------------
    # Training procedure
    # ------------------------------------------------------------
    def update_weights(self, input_vector):
        """
        Update BMU and neighboring neurons using a 2D Gaussian SOM neighborhood.

        Standard SOM update:
            w_i(t+1) = w_i(t) + alpha(t) * h_ci(t) * (x - w_i(t))

        where h_ci(t) depends on 2D grid distance, not feature-space distance.
        """

        lr = self.current_learning_rate
        radius = self.current_radius

        # Safety: avoid division by zero
        radius = max(radius, 1e-8)

        for neuron_index in range(self.num_neurons):

            grid_dist = self.grid_distance_2d(neuron_index, self.winner_idx)

            # Gaussian neighborhood function
            neighborhood_influence = np.exp(-(grid_dist**2) / (2 * radius**2))

            self.neuron_weights[neuron_index] += (
                lr
                * neighborhood_influence
                * (input_vector - self.neuron_weights[neuron_index])
            )

    def kohonen_neighborhood(self):
        """
        Return 2D SOM grid distances from winner to every neuron.
        """

        neighboring_neurons = {}

        winner_row, winner_col = self.index_to_2d(self.winner_idx)

        for neuron_index in range(self.num_neurons):
            row, col = self.index_to_2d(neuron_index)

            distance_from_winner = np.sqrt(
                (row - winner_row) ** 2 + (col - winner_col) ** 2
            )

            neighboring_neurons[neuron_index] = distance_from_winner

        return neighboring_neurons

    def train(self, num_of_epochs, convergence_threshold=1e-3, patience=10):
        """
        Train a 2D SOM using:
            - shuffled input samples
            - decaying learning rate
            - decaying neighborhood radius
            - Gaussian 2D neighborhood update
            - normalized convergence checking
        """
        self.all_steps = num_of_epochs * len(self.input_data)
        self.convergence_counter = 0
        self.epoch_weights = []

        # Store initial weights
        self.epoch_weights.append(np.copy(self.neuron_weights))

        initial_learning_rate = self.learning_rate
        initial_radius = self.radius

        for epoch_index in range(num_of_epochs):

            print(f"Training Epoch {epoch_index + 1}/{num_of_epochs}")

            # -----------------------------
            # Decay learning rate and radius
            # -----------------------------
            self.current_learning_rate = initial_learning_rate * np.exp(
                -epoch_index / num_of_epochs
            )

            self.current_radius = initial_radius * np.exp(-epoch_index / num_of_epochs)

            # Optional: prevent radius from becoming too tiny
            self.current_radius = max(self.current_radius, 1e-3)

            # -----------------------------
            # Shuffle data
            # -----------------------------
            self.input_data = shuffle(self.input_data)

            self.winners_list = []

            # -----------------------------
            # Train one epoch
            # -----------------------------
            for input_vector in self.input_data:

                self.distance = self.calculate_distances(
                    input_vector, self.neuron_weights
                )

                self.winner_idx, _ = self.find_winner_neuron()

                self.winners_list.append(self.winner_idx)

                self.update_weights(input_vector)

            # -----------------------------
            # Compute training metrics
            # -----------------------------
            self.calculate_quantization_error()
            self.calculate_topographic_error()

            # -----------------------------
            # Store weights BEFORE convergence check
            # -----------------------------
            self.epoch_weights.append(np.copy(self.neuron_weights))

            # -----------------------------
            # Check convergence
            # -----------------------------
            if self.check_convergence(epoch_index, convergence_threshold, patience):
                print("Early stopping due to convergence.")
                break

        # Visualize training results
        self.compute_weight_pca_trajectory()
        self.visualize_convergence()
        self.visualize_weight_pca()
        self.plot_weight_changes()
        self.plot_train()
        self.plot_training_error()

        # quantization error plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.error, label="Training Error")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title(f"Training Error Over Time | SOM {self.num_neurons} Neurons")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"figures_new_SOM_{self.num_neurons}/training_error.png")

    def plot_training_error(self):
        """
        Plot quantization error over training epochs.
        """
        if len(self.error) == 0:
            print("No training error values to plot.")
            return

        plt.figure(figsize=(8, 6))

        plt.plot(
            range(1, len(self.error) + 1),
            self.error,
            marker="o",
            label="Quantization Error",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Quantization Error")
        plt.title(
            f"Training Quantization Error Over Epochs | SOM {self.num_neurons} Neurons"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/training_error.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    def calculate_quantization_error(self):
        """
        Compute average quantization error over all input samples.
        """

        squared_bmu_errors = [
            min(self.calculate_distances(input_vector, self.neuron_weights)) ** 2
            for input_vector in self.input_data
        ]

        mean_quantization_error = np.mean(squared_bmu_errors)

        self.error.append(mean_quantization_error)

    def calculate_topographic_error(self):
        """
        Compute Topographic Error (TE) for 2D SOM.
        """

        violations = []

        for input_vector in self.input_data:
            # distances from sample to all neurons
            distances = self.calculate_distances(input_vector, self.neuron_weights)

            # first and second BMUs
            sorted_neuron_indices = np.argsort(distances)

            bmu = sorted_neuron_indices[0]
            second_bmu = sorted_neuron_indices[1]

            # topology preserved?
            if self.are_neighbours(bmu, second_bmu):
                violations.append(0)
            else:
                violations.append(1)

        topographic_error = np.mean(violations)

        self.topographic_error.append(topographic_error)

    def are_neighbours(self, neuron1, neuron2):
        """
        Check if two neurons are neighbors in 2D SOM grid (4-connectivity).
        """

        r1, c1 = self.index_to_2d(neuron1)
        r2, c2 = self.index_to_2d(neuron2)

        return abs(r1 - r2) + abs(c1 - c2) == 1

    def index_to_2d(self, index):
        _, cols = self.grid_shape
        return index // cols, index % cols

    def grid_distance_2d(self, neuron_index, winner_index):
        r1, c1 = self.index_to_2d(neuron_index)
        r2, c2 = self.index_to_2d(winner_index)

        return np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

    def get_cluster_colors(self, cluster_labels):
        """
        Create a stable color mapping for cluster IDs.
        This function assigns a reproducible color to each
        SOM cluster/neuron for visualization.
        """
        unique_cluster_ids = np.unique(cluster_labels)

        color_map = plt.get_cmap("tab20", max(len(unique_cluster_ids), 1))

        cluster_color_lookup = {
            cluster_id: color_map(color_index)
            for color_index, cluster_id in enumerate(unique_cluster_ids)
        }

        return cluster_color_lookup

    def filter_short_bmu_runs(self, bmu_sequence, input_samples, min_run_length=10):
        """
        Keep only consecutive BMU runs whose length >= min_run_length.
        This function removes short-lived noisy BMU state changes and
        preserves only stable posture-cluster segments.
        It's logical that a posture state must persist long enough to
        be meaningful.
        This is a post-processing step for:
            - temporal smoothing
            - noise suppression
            - cluster confidence filtering

        Returns:
            filtered_bmus
            filtered_samples
        """
        bmu_sequence = np.asarray(bmu_sequence)
        input_samples = np.asarray(input_samples)

        if len(bmu_sequence) == 0:
            return bmu_sequence, input_samples

        indices_to_keep = []
        current_run_start = 0

        for sample_index in range(1, len(bmu_sequence)):

            if bmu_sequence[sample_index] != bmu_sequence[sample_index - 1]:
                run_length = sample_index - current_run_start

                if run_length >= min_run_length:
                    indices_to_keep.extend(range(current_run_start, sample_index))

                current_run_start = sample_index

        # Handle final run
        run_length = len(bmu_sequence) - current_run_start

        if run_length >= min_run_length:
            indices_to_keep.extend(range(current_run_start, len(bmu_sequence)))

        indices_to_keep = np.asarray(indices_to_keep, dtype=int)

        return (bmu_sequence[indices_to_keep], input_samples[indices_to_keep])

    def plot_winning_clusters_2d(
        self,
        input_samples,
        winner_labels,
        title,
        save_path=None,
        max_points=None,
        overlay_weights=True,
        alpha=0.7,
        marker_size=14,
    ):
        input_samples = np.asarray(input_samples)
        winner_labels = np.asarray(winner_labels)

        if len(input_samples) == 0:
            print("Nothing to plot (empty input_samples).")
            return

        # Optionally subsample for faster plotting
        if max_points is not None and len(input_samples) > max_points:
            selected_indices = np.random.choice(
                len(input_samples), size=max_points, replace=False
            )
            plotted_samples = input_samples[selected_indices]
            plotted_winners = winner_labels[selected_indices]
        else:
            plotted_samples = input_samples
            plotted_winners = winner_labels

        # PCA projection
        pca = PCA(n_components=2)

        if overlay_weights:
            combined_vectors = np.vstack([plotted_samples, self.neuron_weights])

            projected_vectors = pca.fit_transform(combined_vectors)

            projected_samples = projected_vectors[: len(plotted_samples)]
            projected_weights = projected_vectors[len(plotted_samples) :]
        else:
            projected_samples = pca.fit_transform(plotted_samples)
            projected_weights = None

        colors = self.get_cluster_colors(plotted_winners)

        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(right=0.75)

        for cluster_id in np.unique(plotted_winners):
            cluster_indices = np.where(plotted_winners == cluster_id)[0]

            plt.scatter(
                projected_samples[cluster_indices, 0],
                projected_samples[cluster_indices, 1],
                color=colors[cluster_id],
                label=f"Neuron {cluster_id}",
                alpha=alpha,
                s=marker_size,
                edgecolors="none",
            )

        if overlay_weights and projected_weights is not None:
            plt.scatter(
                projected_weights[:, 0],
                projected_weights[:, 1],
                marker="x",
                s=80,
                linewidths=1.5,
                color="black",
                label="Neuron weights",
            )

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Winners")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------
    # Convergence visualization
    # ------------------------------------------------------------
    def check_convergence(self, epoch_index, convergence_threshold=1e-3, patience=10):
        """
        Stop training if normalized SOM weight changes remain below
        threshold for several consecutive epochs.

        Uses relative (scale-independent) weight change:
            ||W_t - W_{t-1}|| / ||W_{t-1}||
        """

        if len(self.epoch_weights) > 1:

            w_t = self.epoch_weights[-1]
            w_prev = self.epoch_weights[-2]

            # -----------------------------
            # Normalized weight change
            # -----------------------------
            denom = np.linalg.norm(w_prev)

            # Avoid division by zero
            if denom == 0:
                weight_change = 0.0
            else:
                weight_change = np.linalg.norm(w_t - w_prev) / denom

            # -----------------------------
            # Convergence logic
            # -----------------------------
            if weight_change < convergence_threshold:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0

            # -----------------------------
            # Early stopping condition
            # -----------------------------
            if self.convergence_counter >= patience:
                print(
                    f"Converged at epoch {epoch_index} "
                    f"(ΔW = {weight_change:.6e}, patience reached)"
                )
                return True

        return False

    def compute_weight_pca_trajectory(self):
        """
        Fit PCA once after training using all saved epoch weights.
        """

        if len(self.epoch_weights) < 2:
            return

        weight_history_matrix = np.array(self.epoch_weights).reshape(
            len(self.epoch_weights), -1
        )

        pca = PCA(n_components=2)

        self.pca_results = pca.fit_transform(weight_history_matrix)

    def visualize_convergence(self):
        """
        Visualize evolution of SOM weights in PCA space over epochs.
        """

        if len(self.pca_results) < 2:
            return

        pca_trajectory = np.array(self.pca_results)

        epoch_indices = range(1, len(pca_trajectory) + 1)

        plt.figure(figsize=(8, 6))

        plt.plot(epoch_indices, pca_trajectory[:, 0], label="PCA Component 1")

        plt.plot(epoch_indices, pca_trajectory[:, 1], label="PCA Component 2")

        plt.xlabel("Epoch")
        plt.ylabel("PCA Coordinate")
        plt.title(
            f"Convergence of SOM Weights (PCA Space) | SOM {self.num_neurons} Neurons"
        )

        plt.grid(True)
        plt.legend()

        plt.savefig(f"figures_new_SOM_{self.num_neurons}/convergence_pca.png")
        plt.close()

    def visualize_weight_pca(self):
        """
        Visualize trajectory of SOM weight evolution
        projected into a fixed PCA space.
        """

        if len(self.epoch_weights) < 2:
            return

        # Each epoch's full neuron weight matrix becomes one point
        weight_history_matrix = np.array(self.epoch_weights).reshape(
            len(self.epoch_weights), -1
        )

        pca = PCA(n_components=2)

        projected_weight_trajectory = pca.fit_transform(weight_history_matrix)

        plt.figure(figsize=(8, 6))

        # Plot full trajectory
        plt.plot(
            projected_weight_trajectory[:, 0],
            projected_weight_trajectory[:, 1],
            marker="o",
            label="Weight trajectory",
        )

        # Label epoch order
        for epoch_index, (x_coord, y_coord) in enumerate(projected_weight_trajectory):
            plt.text(x_coord, y_coord, str(epoch_index + 1), fontsize=8)

        # Start marker
        plt.scatter(
            projected_weight_trajectory[0, 0],
            projected_weight_trajectory[0, 1],
            marker="s",
            s=100,
            label="Start",
        )

        # End marker
        plt.scatter(
            projected_weight_trajectory[-1, 0],
            projected_weight_trajectory[-1, 1],
            marker="*",
            s=150,
            label="End / Converged",
        )

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(
            f"PCA Trajectory of SOM Weight Evolution | SOM {self.num_neurons} Neurons"
        )

        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/weight_pca.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    # ------------------------------------------------------------
    # Classification (Discriminant Score Evaluation)
    # ------------------------------------------------------------
    def classify(
        self,
        test_samples,
        min_run_length=10,
        apply_filter=True,
        plot_results=True,
        overlay_weights=True,
        max_points=8000,
    ):
        """
        Classify test samples using the BMU and second BMU.

        Optionally filters short BMU runs and plots a PCA scatter
        colored by winning neuron.
        """
        test_samples = np.asarray(test_samples)

        bmu_indices = []
        second_bmu_indices = []

        for input_vector in test_samples:
            distances_to_neurons = self.calculate_distances(
                input_vector, self.neuron_weights
            )

            sorted_neuron_indices = np.argsort(distances_to_neurons)

            bmu_indices.append(sorted_neuron_indices[0])
            second_bmu_indices.append(sorted_neuron_indices[1])

        bmu_indices = np.asarray(bmu_indices)
        second_bmu_indices = np.asarray(second_bmu_indices)

        samples_for_plot = test_samples
        bmus_for_plot = bmu_indices

        if apply_filter:
            bmus_for_plot, samples_for_plot = self.filter_short_bmu_runs(
                bmu_indices, test_samples, min_run_length=min_run_length
            )

            if len(samples_for_plot) == 0:
                print("No clusters survived filtering.")
                return bmu_indices, second_bmu_indices

        if plot_results:
            self.plot_winning_clusters_2d(
                samples_for_plot,
                bmus_for_plot,
                title=(
                    "Winning Neurons (BMU) on Test Data (PCA 2D) | "
                    f"SOM {self.num_neurons} Neurons"
                ),
                save_path=(
                    f"figures_new_SOM_{self.num_neurons}/"
                    "winning_clusters_bmu_pca.png"
                ),
                max_points=max_points,
                overlay_weights=overlay_weights,
            )

            print(
                f"figures_new_SOM_{self.num_neurons}/"
                "winning_clusters_bmu_pca.png, plotted"
            )

        return bmu_indices, second_bmu_indices

    def classify_with_matches(self, test_samples):
        """
        For each test sample, return BMU and 2nd BMU information.
        """

        test_samples = np.asarray(test_samples)

        matching_results = []

        for sample_index, input_vector in enumerate(test_samples):

            distances = self.calculate_distances(input_vector, self.neuron_weights)

            sorted_neuron_indices = np.argsort(distances)

            bmu_index = sorted_neuron_indices[0]
            second_bmu_index = sorted_neuron_indices[1]

            matching_results.append(
                {
                    "sample_index": sample_index,
                    "input_vector": input_vector,
                    "bmu_index": bmu_index,
                    "bmu_distance": distances[bmu_index],
                    "second_bmu_index": second_bmu_index,
                    "second_bmu_distance": distances[second_bmu_index],
                    "bmu_gap": distances[second_bmu_index] - distances[bmu_index],
                }
            )

        return matching_results

    def plot_posture_match(
        self, test_samples, sample_index, save_dir, joint_connections=None
    ):
        """
        Plot one test posture, its BMU prototype, and its 2nd BMU prototype
        as 3D skeletons.
        """
        input_vector = np.asarray(test_samples[sample_index], dtype=float)

        if input_vector.size % 3 != 0:
            raise ValueError(
                f"Input vector length must be divisible by 3, got {input_vector.size}"
            )

        # --------------------------------------------------------
        # Compute BMU and 2nd BMU
        # --------------------------------------------------------
        distances = self.calculate_distances(input_vector, self.neuron_weights)
        sorted_neuron_indices = np.argsort(distances)
        bmu_index = sorted_neuron_indices[0]
        second_bmu_index = sorted_neuron_indices[1]

        # --------------------------------------------------------
        # Reshape to 3D joint coordinates
        # --------------------------------------------------------
        test_posture = input_vector.reshape(-1, 3)
        bmu_posture = self.neuron_weights[bmu_index].reshape(-1, 3)
        second_bmu_posture = self.neuron_weights[second_bmu_index].reshape(-1, 3)

        postures = [
            test_posture,
            bmu_posture,
            second_bmu_posture,
        ]

        titles = [
            f"Test Sample {sample_index}",
            f"BMU {bmu_index}\nDistance = {distances[bmu_index]:.4f}",
            f"2nd BMU {second_bmu_index}\nDistance = {distances[second_bmu_index]:.4f}",
        ]

        # --------------------------------------------------------
        # Common axis limits for fair visual comparison
        # --------------------------------------------------------
        all_points = np.vstack(postures)

        x_min, y_min, z_min = all_points.min(axis=0)
        x_max, y_max, z_max = all_points.max(axis=0)

        max_range = max(
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )

        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2

        axis_limits = {
            "x": (x_mid - max_range / 2, x_mid + max_range / 2),
            "y": (y_mid - max_range / 2, y_mid + max_range / 2),
            "z": (z_mid - max_range / 2, z_mid + max_range / 2),
        }

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------
        fig = plt.figure(figsize=(18, 6))

        for i, (posture, title) in enumerate(zip(postures, titles), start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")

            self.plot_skeleton_3d(
                ax=ax,
                joints=posture,
                joint_connections=joint_connections,
                title=title,
                axis_limits=axis_limits,
            )

        fig.suptitle(
            f"3D Posture Match | SOM {self.num_neurons} Neurons",
            fontsize=14,
        )

        save_path = os.path.join(
            save_dir,
            f"posture_match_sample_{self.num_neurons}.png",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_skeleton_3d(
        self,
        ax,
        joints,
        joint_connections=None,
        title="",
        axis_limits=None,
    ):
        """
        Plot a 3D skeleton using joint coordinates and joint connections.
        """

        import numpy as np

        joints = np.asarray(joints, dtype=float)

        x = joints[:, 0]
        y = joints[:, 1]
        z = joints[:, 2]

        # --------------------------------------------------------
        # Plot joints
        # --------------------------------------------------------
        ax.scatter(
            x,
            y,
            z,
            s=18,
            color="blue",
            depthshade=True,
        )

        # --------------------------------------------------------
        # Plot skeleton links
        # --------------------------------------------------------
        if joint_connections is not None:
            for joint_a, joint_b in joint_connections:
                if joint_a >= len(joints) or joint_b >= len(joints):
                    continue

                ax.plot(
                    [joints[joint_a, 0], joints[joint_b, 0]],
                    [joints[joint_a, 1], joints[joint_b, 1]],
                    [joints[joint_a, 2], joints[joint_b, 2]],
                    color="blue",
                    linewidth=2,
                    marker="o",
                    markersize=3,
                )

        # --------------------------------------------------------
        # Axis formatting
        # --------------------------------------------------------
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.grid(True)

        if axis_limits is not None:
            ax.set_xlim(axis_limits["x"])
            ax.set_ylim(axis_limits["y"])
            ax.set_zlim(axis_limits["z"])

        # Camera angle: adjust if needed
        ax.view_init(elev=20, azim=-70)

        # Make 3D scale visually equal
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    # ------------------------------------------------------------
    # Supporting Methods
    # ------------------------------------------------------------
    def initialize_distance_normalization_constants(
        self, test_samples, neuron_counts, model_dir
    ):
        """
        Compute per-model normalization constants for multiple trained SOM models.

        For each SOM model, this stores:
            - maximum BMU distance
            - maximum BMU-to-second-BMU gap

        These constants allow discriminant and confidence scores to be normalized
        separately for each SOM size instead of using one global summed value.
        """
        self.model_max_bmu_distances = {}
        self.model_max_gap_distances = {}

        for num_neurons in neuron_counts:
            model_file = f"{model_dir}/SOM_{num_neurons}_cls.pkl"

            max_bmu_distance = self.get_max_bmu_distance(model_file, test_samples)

            max_gap_distance = self.get_max_gap_distance(model_file, test_samples)

            self.model_max_bmu_distances[num_neurons] = max_bmu_distance
            self.model_max_gap_distances[num_neurons] = max_gap_distance

    def calculate_discriminant_score(self, neuron_distances, bmu_index, num_neurons):
        """
        Discriminant score based on BMU distance.
        Higher score = more confident assignment.

        Normalize BMU distance using per-model constant.
        """
        max_dist = self.model_max_bmu_distances[num_neurons]

        if max_dist == 0:
            return 1.0

        return max(1 - (neuron_distances[bmu_index] / max_dist), 0)

    def calculate_discriminant_gap_score(self, neuron_distances, num_neurons):
        """
        Confidence score based on BMU and 2nd BMU distance gap.
        Larger gap means more confident BMU assignment.

        Normalize BMU gap using per-model constant.
        """
        if len(neuron_distances) < 2:
            return 0.0

        sorted_distances = np.sort(neuron_distances)
        gap = sorted_distances[1] - sorted_distances[0]
        max_gap = self.model_max_gap_distances[num_neurons]

        if max_gap == 0:
            return 0.0

        return gap / max_gap

    def get_max_bmu_distance(self, model_file, test_samples):
        """
        Maximum BMU distance over all test samples
        for a trained SOM model.
        """

        with open(model_file, "rb") as file_handle:
            model_data = pickle.load(file_handle)

        neuron_weights = model_data["neuron_weights"]

        max_bmu_distance = 0.0

        for sample in test_samples:
            neuron_distances = self.calculate_distances(sample, neuron_weights)
            bmu_distance = min(neuron_distances)
            max_bmu_distance = max(max_bmu_distance, bmu_distance)

        return max_bmu_distance

    def get_max_gap_distance(self, model_file, test_samples):
        """Maximum BMU–2ndBMU gap per model."""
        with open(model_file, "rb") as f:
            model_dict = pickle.load(f)

        neuron_weights = model_dict["neuron_weights"]
        max_gap = 0.0

        for sample in test_samples:
            neuron_distances = self.calculate_distances(sample, neuron_weights)
            sorted_distances = np.sort(neuron_distances)
            gap = sorted_distances[1] - sorted_distances[0]
            max_gap = max(max_gap, gap)

        return max_gap

    # ------------------------------------------------------------
    # Visualization Helpers
    # ------------------------------------------------------------
    def plot_discriminant_series(
        self,
        bmu_sequence,
        discriminant_scores,
        cluster_colors,
        plot_title,
        top_k=4,
        min_segment_length=8,
        downsample_step=1,
        linewidth=1.8,
    ):
        """
        Plots discriminant-score segments colored by BMU cluster.
        Short noisy BMU runs are ignored to reduce visual clutter.
        """
        bmu_sequence = np.asarray(bmu_sequence)
        discriminant_scores = np.asarray(discriminant_scores)

        if plot_title is None:
            plot_title = f"Discriminant Score Series | SOM {self.num_neurons} neurons"

        # --------------------------------------------------------
        # Select clusters to show
        # --------------------------------------------------------
        unique_bmus, counts = np.unique(bmu_sequence, return_counts=True)

        if top_k is not None:
            sorted_idx = np.argsort(counts)[::-1]
            visible_clusters = unique_bmus[sorted_idx[:top_k]]
        else:
            visible_clusters = unique_bmus

        visible_clusters = set(visible_clusters)

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------
        plt.figure(figsize=(10, 6))
        labeled_clusters = set()

        # --------------------------------------------------------
        # Plot clean continuous segments
        # --------------------------------------------------------
        for cluster_id in sorted(visible_clusters):

            active_indices = np.where(bmu_sequence == cluster_id)[0]

            if len(active_indices) == 0:
                continue

            breaks = np.where(np.diff(active_indices) > 1)[0]

            segment_ranges = zip(
                np.r_[0, breaks + 1],
                np.r_[breaks, len(active_indices) - 1],
            )

            for start_idx, end_idx in segment_ranges:
                segment = active_indices[start_idx:end_idx + 1]

                if len(segment) < min_segment_length:
                    continue

                segment = segment[::downsample_step]

                if len(segment) < 2:
                    continue

                plt.plot(
                    segment,
                    discriminant_scores[segment],
                    color=cluster_colors[cluster_id],
                    linewidth=linewidth,
                    alpha=0.9,
                )

                if cluster_id not in labeled_clusters:
                    plt.plot(
                        [],
                        [],
                        color=cluster_colors[cluster_id],
                        linewidth=linewidth,
                        label=f"Cluster {cluster_id}",
                    )
                    labeled_clusters.add(cluster_id)

        # --------------------------------------------------------
        # Formatting
        # --------------------------------------------------------
        plt.xlabel("Sample Index")
        plt.ylabel("Discriminant Score")
        plt.title(plot_title)

        plt.grid(True, alpha=0.25)

        if labeled_clusters:
            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
            )

        plt.tight_layout()

        # --------------------------------------------------------
        # Save
        # --------------------------------------------------------
        save_dir = f"figures_new_SOM_{self.num_neurons}"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "discriminant_series_clean.png")

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_discriminant_subplots(
        self,
        bmu_sequence,
        input_samples,
        cluster_colors,
        num_neurons,
        top_k=4,
        smooth_window=15,
        min_segment_length=8,
        shade_alpha=0.18,
    ):
        """
        Plot discriminant-score trends for selected winning neurons.

        Each subplot corresponds to one BMU/neuron.
        Shaded regions show where that neuron is the active winner.
        """
        bmu_sequence = np.asarray(bmu_sequence)
        input_samples = np.asarray(input_samples)

        # --------------------------------------------------------
        # Select top-k most frequent BMUs
        # --------------------------------------------------------
        unique_bmus, counts = np.unique(bmu_sequence, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        selected_bmus = unique_bmus[sorted_indices[:top_k]]

        # --------------------------------------------------------
        # Create subplots
        # --------------------------------------------------------
        fig, axes = plt.subplots(
            len(selected_bmus),
            1,
            figsize=(14, 4 * len(selected_bmus)),
            sharex=True,
        )

        if len(selected_bmus) == 1:
            axes = [axes]

        fig.suptitle(
            f"Discriminant Scores for Top {top_k} Winning Neurons | "
            f"SOM {self.num_neurons} Neurons",
            fontsize=14,
        )

        # --------------------------------------------------------
        # Plot each selected BMU/neuron
        # --------------------------------------------------------
        for subplot_index, neuron_index in enumerate(selected_bmus):

            ax = axes[subplot_index]

            neuron_discriminant_scores = []

            for input_vector in input_samples:
                neuron_distances = self.calculate_distances(
                    input_vector,
                    self.neuron_weights,
                )

                score = self.calculate_discriminant_score(
                    neuron_distances,
                    neuron_index,
                    num_neurons,
                )

                neuron_discriminant_scores.append(score)

            neuron_discriminant_scores = np.asarray(neuron_discriminant_scores)

            # Smooth curve
            smoothed_scores = self.moving_average(
                neuron_discriminant_scores,
                smooth_window,
            )

            x = np.arange(len(smoothed_scores))

            # ----------------------------------------------------
            # Plot discriminant curve
            # ----------------------------------------------------
            ax.plot(
                x,
                smoothed_scores,
                color=cluster_colors[neuron_index],
                linewidth=1.5,
                alpha=0.95,
            )

            # ----------------------------------------------------
            # Shade active BMU regions
            # ----------------------------------------------------
            winning_sample_indices = np.where(bmu_sequence == neuron_index)[0]

            if len(winning_sample_indices) > 0:

                breaks = np.where(np.diff(winning_sample_indices) > 1)[0]

                segment_ranges = zip(
                    np.r_[0, breaks + 1],
                    np.r_[breaks, len(winning_sample_indices) - 1],
                )

                for start_idx, end_idx in segment_ranges:
                    active_region = winning_sample_indices[start_idx:end_idx + 1]

                    # Ignore very short noisy BMU activations
                    if len(active_region) < min_segment_length:
                        continue

                    ax.axvspan(
                        active_region[0],
                        active_region[-1],
                        color=cluster_colors[neuron_index],
                        alpha=shade_alpha,
                    )

            # ----------------------------------------------------
            # Formatting
            # ----------------------------------------------------
            ax.set_title(f"Neuron {neuron_index} Discriminant Scores")
            ax.set_ylabel("Discriminant")
            ax.set_xlabel("Sample Index")
            ax.grid(True, alpha=0.35)

            # Optional: keep y-axis tight but clean
            y_min = np.nanmin(smoothed_scores)
            y_max = np.nanmax(smoothed_scores)
            margin = 0.05 * (y_max - y_min + 1e-8)

            ax.set_ylim(
                max(0.0, y_min - margin),
                min(1.05, y_max + margin),
            )

        # --------------------------------------------------------
        # Save figure
        # --------------------------------------------------------
        save_dir = f"figures_new_SOM_{self.num_neurons}"
        os.makedirs(save_dir, exist_ok=True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = os.path.join(
            save_dir,
            "discriminant_subplots.png",
        )
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    # ------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------
    def plot_weight_changes(self, convergence_threshold=0.05):
        """
        Plot L2 norm of weight differences between consecutive epochs.
        This reflects how much SOM prototypes move during training.

        --Convergence Threshold line--
        This adds a visual boundary showing when SOM weight updates
        become small enough to be considered converged.
        """

        if len(self.epoch_weights) < 2:
            print("Not enough epochs to compute weight changes.")
            return

        weight_change_norms = [
            np.linalg.norm(
                self.epoch_weights[epoch_idx + 1] - self.epoch_weights[epoch_idx]
            )
            for epoch_idx in range(len(self.epoch_weights) - 1)
        ]

        epoch_indices = range(1, len(weight_change_norms) + 1)

        plt.figure(figsize=(8, 6))

        plt.plot(
            epoch_indices,
            weight_change_norms,
            marker="o",
            linewidth=1.5,
            label="Weight Change",
        )

        # Add convergence threshold line if provided
        if convergence_threshold is not None:
            plt.axhline(
                y=convergence_threshold,
                linestyle="--",
                color="red",
                linewidth=1.5,
                label="Convergence Threshold",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Weight Change Δ (L2 Norm)")
        plt.title(
            f"SOM Weight Convergence Over Epochs | SOM {self.num_neurons} Neurons"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/weight_changes.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def plot_train(self, max_points=4000, overlay_weights=True):
        """
        PCA visualization of training data colored by BMU clusters.
        Optionally overlays neuron weight prototypes.
        """

        input_data = np.asarray(self.input_data)
        winners = np.asarray(self.winners_list)

        if len(input_data) == 0:
            print("No data to plot.")
            return

        # Optional subsampling
        if max_points is not None and len(input_data) > max_points:
            idx = np.random.choice(len(input_data), max_points, replace=False)
            input_data = input_data[idx]
            winners = winners[idx]

        # PCA (fit on data + weights for consistency)
        pca = PCA(n_components=2)

        if overlay_weights:
            combined = np.vstack([input_data, self.neuron_weights])
            reduced = pca.fit_transform(combined)
            X2 = reduced[: len(input_data)]
            W2 = reduced[len(input_data) :]
        else:
            X2 = pca.fit_transform(input_data)
            W2 = None

        # Use only active clusters
        unique_clusters = np.unique(winners)

        # Stable colors
        colors = plt.get_cmap("tab20", len(unique_clusters))
        cluster_color_map = {c: colors(i) for i, c in enumerate(unique_clusters)}

        plt.figure(figsize=(10, 8))

        for c in unique_clusters:
            idx = np.where(winners == c)[0]

            plt.scatter(
                X2[idx, 0],
                X2[idx, 1],
                color=cluster_color_map[c],
                label=f"Cluster {c}",
                s=12,
                alpha=0.7,
                edgecolors="none",
            )

        # Overlay neuron weights
        if overlay_weights and W2 is not None:
            plt.scatter(
                W2[:, 0],
                W2[:, 1],
                marker="x",
                s=80,
                linewidths=1.5,
                color="black",
                label="Neuron Prototypes",
            )

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(
            f"Training Clusters (PCA Projection) | SOM {self.num_neurons} Neurons"
        )
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Clusters")
        plt.tight_layout()
        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/train_pca.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def plot_metric_across_models(
        self,
        metric_dict,
        bmu_sequence_dict,
        cluster_colors_dict,
        metric_name,
        ylabel,
        save_dir,
        top_k=4,
        step=25,
        smooth_window=75,
        min_segment_length=120,
        shade_alpha=0.08,
    ):
        """
        Plot one metric trend across input samples for each SOM model.

        Each subplot corresponds to one SOM model.
        Shaded regions show where each neuron is the active BMU inside that model.

        Args:
            metric_dict:
                dict[num_neurons] -> array of metric values per sample

            bmu_sequence_dict:
                dict[num_neurons] -> array of BMU indices per sample

            cluster_colors_dict:
                dict[num_neurons] -> dict/list mapping neuron index to color

            metric_name:
                Name of metric, e.g. "QE", "TE", "Discriminant Score", "Gap Score"

            ylabel:
                Label for y-axis

            save_dir:
                Directory where figure is saved
        """
        model_ids = sorted(metric_dict.keys())
        num_models = len(model_ids)

        fig, axes = plt.subplots(
            num_models,
            1,
            figsize=(14, 3.2 * num_models),
            sharex=True,
        )

        if num_models == 1:
            axes = [axes]

        fig.suptitle(
            f"{metric_name} Trends Across Input Samples for All SOM Models",
            fontsize=16,
        )

        for subplot_index, num_neurons in enumerate(model_ids):
            ax = axes[subplot_index]

            metric_values = np.asarray(metric_dict[num_neurons], dtype=float)
            bmu_sequence = np.asarray(bmu_sequence_dict[num_neurons])
            cluster_colors = cluster_colors_dict[num_neurons]

            full_x = np.arange(len(metric_values))

            # =====================================================
            # 1. Smooth + downsample blue metric line
            # =====================================================
            smoothed_metric = self.moving_average(metric_values, smooth_window)

            x_plot = full_x[::step]
            y_plot = smoothed_metric[::step]

            ax.plot(
                x_plot,
                y_plot,
                linewidth=1.4,
                label=f"SOM {num_neurons}",
            )

            # =====================================================
            # 2. Select top-k most active BMU neurons
            # =====================================================
            unique, counts = np.unique(bmu_sequence, return_counts=True)

            sorted_indices = np.argsort(counts)[::-1]
            top_neurons = unique[sorted_indices[:top_k]]

            # =====================================================
            # 3. Shade only long active regions of top neurons
            # =====================================================
            for neuron_index in top_neurons:
                winning_sample_indices = np.where(bmu_sequence == neuron_index)[0]

                if len(winning_sample_indices) == 0:
                    continue

                breaks = np.where(np.diff(winning_sample_indices) > 1)[0]

                segment_ranges = zip(
                    np.r_[0, breaks + 1],
                    np.r_[breaks, len(winning_sample_indices) - 1],
                )

                for start_idx, end_idx in segment_ranges:
                    active_region = winning_sample_indices[start_idx : end_idx + 1]

                    if len(active_region) < min_segment_length:
                        continue

                    ax.axvspan(
                        active_region[0],
                        active_region[-1],
                        color=cluster_colors[neuron_index],
                        alpha=shade_alpha,
                    )

            ax.set_title(f"SOM {num_neurons} Neurons | {metric_name}")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Input Sample Index")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        filename = f"{metric_name.lower().replace(' ', '_')}_trends_by_model.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_training_quantization_error(self):
        """
        Plot Quantization Error (QE) over SOM training epochs.

        QE measures the average distance between each training sample
        and its Best Matching Unit (BMU). Lower QE indicates that the
        SOM prototypes better represent the training data distribution.
        """
        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------
        if len(self.error) == 0:
            print("No quantization error values found.")
            return

        # --------------------------------------------------------
        # Epoch indices
        # --------------------------------------------------------
        epochs = np.arange(1, len(self.error) + 1)

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------
        plt.figure(figsize=(10, 5))

        plt.plot(
            epochs,
            self.error,
            marker="o",
            linewidth=2,
            markersize=4,
            label="Training QE",
        )

        # --------------------------------------------------------
        # Optional smoothing
        # --------------------------------------------------------
        if len(self.error) > 5:
            smooth_window = min(10, len(self.error))
            kernel = np.ones(smooth_window) / smooth_window
            smoothed_qe = np.convolve(
                self.error,
                kernel,
                mode="same",
            )

            plt.plot(
                epochs,
                smoothed_qe,
                linewidth=2.5,
                linestyle="--",
                label="Smoothed QE",
            )

        # --------------------------------------------------------
        # Highlight minimum QE epoch
        # --------------------------------------------------------
        min_qe_index = np.argmin(self.error)

        plt.scatter(
            epochs[min_qe_index],
            self.error[min_qe_index],
            s=80,
            marker="*",
            label=f"Minimum QE = {self.error[min_qe_index]:.4f}",
        )

        # --------------------------------------------------------
        # Labels and formatting
        # --------------------------------------------------------
        plt.title(
            f"SOM Training Quantization Error | SOM {self.num_neurons} Neurons"
        )

        plt.xlabel("Training Epoch")
        plt.ylabel("Quantization Error (QE)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # --------------------------------------------------------
        # Save figure
        # --------------------------------------------------------
        save_dir = f"figures_new_SOM_{self.num_neurons}"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            "training_quantization_error.png",
        )
        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def plot_training_topographic_error(self):
        """
        Plot Topographic Error (TE) over SOM training epochs.

        TE measures topology preservation in the SOM.
        It is the fraction of samples for which the
        first and second BMUs are not neighbors.

        Lower TE indicates better topology preservation.
        """
        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------
        if len(self.topographic_error) == 0:
            print("No topographic error values found.")
            return

        # --------------------------------------------------------
        # Epoch indices
        # --------------------------------------------------------
        epochs = np.arange(1, len(self.topographic_error) + 1)

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------
        plt.figure(figsize=(10, 5))

        plt.plot(
            epochs,
            self.topographic_error,
            marker="o",
            linewidth=2,
            markersize=4,
            label="Training TE",
        )

        # --------------------------------------------------------
        # Optional smoothing
        # --------------------------------------------------------
        if len(self.topographic_error) > 5:

            smooth_window = min(10, len(self.topographic_error))

            kernel = np.ones(smooth_window) / smooth_window

            smoothed_te = np.convolve(
                self.topographic_error,
                kernel,
                mode="same",
            )

            plt.plot(
                epochs,
                smoothed_te,
                linewidth=2.5,
                linestyle="--",
                label="Smoothed TE",
            )

        # --------------------------------------------------------
        # Highlight minimum TE epoch
        # --------------------------------------------------------
        min_te_index = np.argmin(self.topographic_error)

        plt.scatter(
            epochs[min_te_index],
            self.topographic_error[min_te_index],
            s=80,
            marker="*",
            label=f"Minimum TE = {self.topographic_error[min_te_index]:.4f}",
        )

        # --------------------------------------------------------
        # Labels and formatting
        # --------------------------------------------------------
        plt.title(
            f"SOM Training Topographic Error | SOM {self.num_neurons} Neurons"
        )

        plt.xlabel("Training Epoch")
        plt.ylabel("Topographic Error (TE)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # --------------------------------------------------------
        # Save figure
        # --------------------------------------------------------
        save_dir = f"figures_new_SOM_{self.num_neurons}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            "training_topographic_error.png",
        )
        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def plot_training_discriminant_scores(self, smooth_window=5):
        """
        Plot training discriminant scores across SOM training epochs.

        The discriminant score is used here as a cluster-confidence metric,
        indicating how strongly the SOM assigns training samples to their BMUs.

        Higher discriminant scores indicate:
        - stronger cluster assignment
        - better BMU separation
        - more confident SOM representation
        """
        if len(self.epoch_weights) == 0:
            print("No epoch weights found. Train the SOM first.")
            return

        epoch_mean_discriminant_scores = []

        for epoch_weights in self.epoch_weights:

            bmu_distances = []

            # First pass: collect BMU distances for this epoch
            for input_vector in self.input_data:
                neuron_distances = self.calculate_distances(
                    input_vector,
                    epoch_weights
                )

                bmu_distance = np.min(neuron_distances)
                bmu_distances.append(bmu_distance)

            max_bmu_distance = np.max(bmu_distances)

            # Second pass: compute normalized discriminant scores
            epoch_scores = []

            for bmu_distance in bmu_distances:
                if max_bmu_distance == 0:
                    score = 1.0
                else:
                    score = 1 - (bmu_distance / max_bmu_distance)

                score = np.clip(score, 0.0, 1.0)
                epoch_scores.append(score)

            epoch_mean_discriminant_scores.append(np.mean(epoch_scores))

        epoch_mean_discriminant_scores = np.asarray(epoch_mean_discriminant_scores)

        epochs = np.arange(1, len(epoch_mean_discriminant_scores) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(
            epochs,
            epoch_mean_discriminant_scores,
            marker="o",
            linewidth=2,
            markersize=4,
            label="Mean Training Discriminant Score",
        )

        if len(epoch_mean_discriminant_scores) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed_scores = np.convolve(
                epoch_mean_discriminant_scores,
                kernel,
                mode="same"
            )

            plt.plot(
                epochs,
                smoothed_scores,
                linestyle="--",
                linewidth=2.5,
                label="Smoothed Trend",
            )

        max_idx = np.argmax(epoch_mean_discriminant_scores)

        plt.scatter(
            epochs[max_idx],
            epoch_mean_discriminant_scores[max_idx],
            s=90,
            marker="*",
            label=f"Max = {epoch_mean_discriminant_scores[max_idx]:.4f}",
        )

        plt.title(
            f"SOM Training Discriminant Score | SOM {self.num_neurons} Neurons"
        )
        plt.xlabel("Training Epoch")
        plt.ylabel("Mean Discriminant Score")
        plt.grid(True, alpha=0.3)
        plt.legend()

        save_dir = f"figures_new_SOM_{self.num_neurons}"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            "training_discriminant_scores.png"
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def save(self, filename):
        """
        Save model safely, handling both:
        - filenames with directories
        - filenames without directories
        """

        directory = os.path.dirname(filename)

        # Only create directory if it exists in path
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename="models"):
        with open(filename, "rb") as f:
            state_dict = pickle.load(f)

        self.__dict__.clear()
        self.__dict__.update(state_dict)

    def moving_average(self, values, window):
        values = np.asarray(values, dtype=float)

        if window <= 1 or len(values) < window:
            return values

        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="same")


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    training_data = "training_data"
    test_data = "test_data"

    # Define models - [6 - 20]
    neuron_range = range(6, 22, 2)

    learning_rate = 0.2
    dist_metric = "cosine"
    num_epochs = 100
    convergence_threshold = 1e-3
    patience = 10

    model_dir = "trained_som_models"
    decision_dir = "model_decision"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(decision_dir, exist_ok=True)

    results = []

    # ------------------------------------------------------------
    # Train SOM models from 6 to 20 neurons
    # ------------------------------------------------------------
    for num_neurons in neuron_range:
        print("\n================================================")
        print(f"Training SOM with {num_neurons} neurons")
        print("==================================================")

        os.makedirs(f"figures_new_SOM_{num_neurons}", exist_ok=True)
        rows, cols = compute_grid_shape(num_neurons)

        som = CompetitiveLearning(
            num_neurons=num_neurons,
            training_data=training_data,
            radius=(max(rows, cols) / 2),
            learning_rate=learning_rate,
            gaussian=True,
            dist_metric=dist_metric,
        )

        som.train(
            num_of_epochs=num_epochs,
            convergence_threshold=convergence_threshold,
            patience=patience,
        )
        som.plot_training_quantization_error()
        som.plot_training_topographic_error()
        som.plot_training_discriminant_scores()

        model_path = f"{model_dir}/SOM_{num_neurons}_cls.pkl"
        som.save(model_path)

    # ------------------------------------------------------------
    # Test SOM models from 6 to 20 neurons
    # ------------------------------------------------------------
    # Load test data
    test_samples = load_test_data(test_data)

    comparative_metrics = {
        "QE": {},
        "TE": {},
        "discriminant": {},
        "gap": {},
    }

    bmu_sequences_by_model = {}
    cluster_colors_by_model = {}

    comparison_dir = "model_comparison_plots"
    os.makedirs(comparison_dir, exist_ok=True)

    postures_save_dir = "figures_posture_match"
    os.makedirs(postures_save_dir, exist_ok=True)

    for num_neurons in neuron_range:
        # --------------------------------------------------------
        # Load trained model
        # --------------------------------------------------------
        rows, cols = compute_grid_shape(num_neurons)
        loaded_som = CompetitiveLearning(
            num_neurons=num_neurons,
            training_data=training_data,
            radius=(max(rows, cols) / 2),
            learning_rate=learning_rate,
            gaussian=True,
            dist_metric=dist_metric,
        )

        model_path = f"{model_dir}/SOM_{num_neurons}_cls.pkl"

        loaded_som.load(model_path)

        # Initialize normalization
        loaded_som.initialize_distance_normalization_constants(
            test_samples, neuron_range, model_dir
        )

        # --------------------------------------------------------
        # Classify test data
        # --------------------------------------------------------
        bmu_sequence, second_bmu_sequence = loaded_som.classify(
            test_samples
        )

        # --------------------------------------------------------
        # Discriminant scores and gap scores
        # --------------------------------------------------------
        discriminant_scores = []
        gap_scores = []

        # --------------------------------------------------------
        # Per-sample metric containers for this SOM model
        # --------------------------------------------------------
        sample_qe_values = []
        sample_te_values = []
        sample_discriminant_scores = []
        sample_gap_scores = []

        for sample in test_samples:
            neuron_distances = loaded_som.calculate_distances(
                sample, loaded_som.neuron_weights
            )

            sorted_indices = np.argsort(neuron_distances)
            bmu_index = sorted_indices[0]
            second_bmu_index = sorted_indices[1]

            discriminant_score = loaded_som.calculate_discriminant_score(
                neuron_distances, bmu_index, loaded_som.num_neurons
            )

            gap_score = loaded_som.calculate_discriminant_gap_score(
                neuron_distances, loaded_som.num_neurons
            )

            discriminant_scores.append(discriminant_score)
            gap_scores.append(gap_score)

            # -----------------------------
            # Per-sample QE
            # -----------------------------
            # QE per sample = distance from sample to BMU
            sample_qe = neuron_distances[bmu_index]

            # -----------------------------
            # Per-sample TE
            # -----------------------------
            # TE per sample = 0 if BMU and 2nd BMU are neighbors, else 1
            if loaded_som.are_neighbours(bmu_index, second_bmu_index):
                sample_te = 0
            else:
                sample_te = 1

            # -----------------------------
            # Discriminant score
            # -----------------------------
            discriminant_score = loaded_som.calculate_discriminant_score(
                neuron_distances, bmu_index, loaded_som.num_neurons
            )

            # -----------------------------
            # Gap score
            # -----------------------------
            gap_score = loaded_som.calculate_discriminant_gap_score(
                neuron_distances, loaded_som.num_neurons
            )

            sample_qe_values.append(sample_qe)
            sample_te_values.append(sample_te)
            sample_discriminant_scores.append(discriminant_score)
            sample_gap_scores.append(gap_score)
            bmu_sequences_by_model[num_neurons] = np.asarray(bmu_sequence)
            cluster_colors_by_model[num_neurons] = loaded_som.get_cluster_colors(
                bmu_sequence
            )

        # Convert to numpy arrays
        sample_qe_values = np.asarray(sample_qe_values)
        sample_te_values = np.asarray(sample_te_values)
        sample_discriminant_scores = np.asarray(sample_discriminant_scores)
        sample_gap_scores = np.asarray(sample_gap_scores)

        # Store for comparative plotting
        comparative_metrics["QE"][num_neurons] = sample_qe_values
        comparative_metrics["TE"][num_neurons] = sample_te_values
        comparative_metrics["discriminant"][num_neurons] = sample_discriminant_scores
        comparative_metrics["gap"][num_neurons] = sample_gap_scores

        discriminant_scores = np.asarray(discriminant_scores)
        gap_scores = np.asarray(gap_scores)

        mean_discriminant = np.mean(discriminant_scores)
        std_discriminant = np.std(discriminant_scores)

        mean_gap_score = np.mean(gap_scores)
        std_gap_score = np.std(gap_scores)

        final_qe = loaded_som.error[-1]
        final_te = loaded_som.topographic_error[-1]

        results.append(
            {
                "neurons": num_neurons,
                "QE": final_qe,
                "TE": final_te,
                "mean_discriminant": mean_discriminant,
                "std_discriminant": std_discriminant,
                "mean_gap_score": mean_gap_score,
                "std_gap_score": std_gap_score,
            }
        )

        # --------------------------------------------------------
        # Plot discriminant series
        # --------------------------------------------------------
        cluster_colors = loaded_som.get_cluster_colors(bmu_sequence)

        loaded_som.plot_discriminant_series(
            bmu_sequence=bmu_sequence,
            discriminant_scores=discriminant_scores,
            cluster_colors=cluster_colors,
            plot_title=f"Discriminant Score Series | SOM {num_neurons} Neurons",
        )

        loaded_som.plot_discriminant_subplots(
            bmu_sequence=bmu_sequence,
            input_samples=test_samples,
            cluster_colors=cluster_colors,
            num_neurons=loaded_som.num_neurons,
        )

        # --------------------------------------------------------
        # Plot posture match example
        # --------------------------------------------------------
        # loaded_som.plot_posture_match(
        #     test_samples=test_samples,
        #     sample_index=0,
        #     save_dir=postures_save_dir,
        #     joint_connections=loaded_som.joint_connections,
        # )

    # --------------------------------------------------------
    # Plot Metrics accross models - QE, TE, DS, GAP
    # --------------------------------------------------------
    loaded_som.plot_metric_across_models(
        metric_dict=comparative_metrics["QE"],
        bmu_sequence_dict=bmu_sequences_by_model,
        cluster_colors_dict=cluster_colors_by_model,
        metric_name="Quantization Error",
        ylabel="QE / BMU Distance",
        save_dir=comparison_dir,
    )

    loaded_som.plot_metric_across_models(
        metric_dict=comparative_metrics["TE"],
        bmu_sequence_dict=bmu_sequences_by_model,
        cluster_colors_dict=cluster_colors_by_model,
        metric_name="Topographic Error",
        ylabel="TE Violation",
        save_dir=comparison_dir,
    )

    loaded_som.plot_metric_across_models(
        metric_dict=comparative_metrics["discriminant"],
        bmu_sequence_dict=bmu_sequences_by_model,
        cluster_colors_dict=cluster_colors_by_model,
        metric_name="Discriminant Score",
        ylabel="Discriminant Score",
        save_dir=comparison_dir,
    )

    loaded_som.plot_metric_across_models(
        metric_dict=comparative_metrics["gap"],
        bmu_sequence_dict=bmu_sequences_by_model,
        cluster_colors_dict=cluster_colors_by_model,
        metric_name="Gap Score",
        ylabel="Normalized Gap Score",
        save_dir=comparison_dir,
    )

    # --------------------------------------------------------
    # Plot QE + TE + Discriminant Stability
    # --------------------------------------------------------
    neurons = [r["neurons"] for r in results]
    qe_values = [r["QE"] for r in results]
    te_values = [r["TE"] for r in results]
    mean_ds = [r["mean_discriminant"] for r in results]
    std_ds = [r["std_discriminant"] for r in results]
    mean_gap = [r["mean_gap_score"] for r in results]
    std_gap = [r["std_gap_score"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(neurons, qe_values, marker="o", label="QE")
    plt.plot(neurons, te_values, marker="o", label="TE")
    plt.plot(neurons, mean_ds, marker="o", label="Mean Discriminant")
    plt.plot(neurons, std_ds, marker="o", label="Std Discriminant")
    plt.plot(neurons, mean_gap, marker="o", label="Mean Gap Score")
    plt.plot(neurons, std_gap, marker="o", label="Std Gap Score")

    plt.xlabel("Number of SOM Neurons")
    plt.ylabel("Score")
    plt.title("SOM Model Selection: QE + TE + Discriminant Stability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(decision_dir, "som_model_selection_metrics.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------------
    # Estimate optimal number of neurons
    # --------------------------------------------------------
    best_model = None
    best_score = float("inf")

    for r in results:
        combined_score = (
            r["QE"]
            + r["TE"]
            + r["std_discriminant"]
            - r["mean_discriminant"]
            - r["mean_gap_score"]
            + r["std_gap_score"]
        )

        if combined_score < best_score:
            best_score = combined_score
            best_model = r

    print("Best SOM model: ")
    print(best_model)


# TO DO
# use U-matrix and cluster boundaries
