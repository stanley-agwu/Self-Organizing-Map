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

    # ------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------
    def calculate_distances(self, input_vector, neuron_weights):
        """
        Compute the distance between an input vector (posture sample) and each neuron weight vector
        using the selected distance metric.
        """

        distance_functions = {
            "manhattan": distance.cityblock,
            "minkowski": lambda x, y: distance.minkowski(x, y, p=3),
            "hamming": distance.hamming,
            "cosine": distance.cosine,
            "euclidean": distance.euclidean,
        }

        # Use chosen metric, default to Euclidean
        distance_function = distance_functions.get(self.dist_metric, distance.euclidean)

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
        Update the BMU and its neighboring neuron weights.
        """

        # Update the winning neuron / BMU
        self.neuron_weights[self.winner_idx] += self.learning_rate * (
            input_vector - self.neuron_weights[self.winner_idx]
        )

        # Update neighboring neurons with smaller influence
        for neuron_index, distance_from_winner in self.kohonen_neighborhood().items():
            if neuron_index == self.winner_idx:
                continue
            if distance_from_winner <= self.radius:
                self.neuron_weights[neuron_index] += (
                    0.8
                    * self.learning_rate
                    * (input_vector - self.neuron_weights[neuron_index])
                )

    def kohonen_neighborhood(self):
        """
        Return neurons whose prototype vectors lie within the neighborhood
        radius of the winning neuron.
        """
        neighboring_neurons = {}
        winner_weight = self.neuron_weights[self.winner_idx]

        for neuron_index, neuron_weight in enumerate(self.neuron_weights):
            distance_to_winner = self.calculate_distances(
                winner_weight, [neuron_weight]
            )[0]

            if distance_to_winner <= self.radius:
                neighboring_neurons[neuron_index] = distance_to_winner

        return neighboring_neurons

    def train(self, num_of_epochs, convergence_threshold=0.1, patience=10):
        """
        Train the SOM network. It repeatedly presents posture samples to the neurons,
        finds the BMU, updates the BMU/neighborhood weights, tracks error, saves models,
        checks convergence, and finally generates plots
        """
        self.all_steps = num_of_epochs * len(self.input_data)

        for epoch_index in range(num_of_epochs):
            print(f"Training Epoch {epoch_index}")

            # Shuffle training samples so the SOM does not learn sequence bias
            self.input_data = shuffle(self.input_data)

            # Store BMU assignments for this epoch
            self.winners_list = []

            for input_vector in self.input_data:

                # Compute distance from current input to all neuron prototypes
                self.distance = self.calculate_distances(
                    input_vector, self.neuron_weights
                )

                # Find Best Matching Unit
                self.winner_idx, _ = self.find_winner_neuron()

                # Store winning neuron index
                self.winners_list.append(self.winner_idx)

                # Move BMU and neighbors toward the input vector
                self.update_weights(input_vector)

            # Compute quantization/training error after epoch
            self.calculate_quantization_error()
            self.calculate_topographic_error()

            # Save checkpoint every 5 epochs
            if (epoch_index + 1) % 5 == 0:

                # Directory per neuron count
                neuron_dir = f"SOM_neuron_{self.num_neurons}"
                os.makedirs(neuron_dir, exist_ok=True)

                # Full path = directory + filename
                save_filename = os.path.join(
                    neuron_dir,
                    f"SOM_neurons_{self.num_neurons}_epoch_{epoch_index + 1}.pkl",
                )

                print(
                    f"Saving model for {self.num_neurons} neurons "
                    f"at epoch {epoch_index + 1} → {save_filename}"
                )

                self.save(save_filename)

            # Stop training if weights have converged
            if self.check_convergence(epoch_index, convergence_threshold, patience):
                print("Early stopping due to convergence.")
                break

            # Store copy of weights for later visualization
            self.epoch_weights.append(np.copy(self.neuron_weights))

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
        plt.title("Training Error Over Time")
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
        plt.title("Training Quantization Error Over Epochs")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/training_error.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show(block=False)
        plt.pause(0.001)

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
        Compute Topographic Error (TE).
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

    # NOTE: If SOM grids changes to 2D, this function must change.
    def are_neighbours(self, neuron1, neuron2):
        """
        Two neurons are neighbors if adjacent in 1D SOM.
        """
        return abs(neuron1 - neuron2) == 1

    def get_cluster_colors(self, cluster_labels):
        """
        Create a stable color mapping for cluster IDs.
        This function assigns a reproducible color to each
        SOM cluster/neuron for visualization.
        """
        unique_cluster_ids = np.unique(cluster_labels)

        color_map = plt.cm.get_cmap("tab20", max(len(unique_cluster_ids), 1))

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
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------
    # Convergence visualization
    # ------------------------------------------------------------
    def check_convergence(self, epoch_index, convergence_threshold, patience):
        """
        Stop training if raw SOM weight changes remain below
        threshold for several consecutive epochs.
        """

        if len(self.epoch_weights) > 1:

            weight_change = np.linalg.norm(
                self.epoch_weights[-1] - self.epoch_weights[-2]
            )

            if weight_change < convergence_threshold:
                self.convergence_counter += 1

                if self.convergence_counter >= patience:
                    return True
            else:
                self.convergence_counter = 0

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
        plt.title("Convergence of SOM Weights (PCA Space)")

        plt.grid(True)
        plt.legend()

        plt.savefig(f"figures_new_SOM_{self.num_neurons}/convergence_pca.png")

        plt.show(block=False)
        plt.pause(0.001)

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
        plt.title("PCA Trajectory of SOM Weight Evolution")

        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/weight_pca.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show(block=False)
        plt.pause(0.001)

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
        max_points=6000,
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
                    f"min_run_len={min_run_length if apply_filter else 0}"
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

    def plot_posture_match(self, test_samples, sample_index, joint_connections=None):
        """
        Plot one test posture, its BMU prototype, and its 2nd BMU prototype.
        Assumes posture vectors are flattened 3D joint coordinates.
        """

        input_vector = test_samples[sample_index]

        distances = self.calculate_distances(input_vector, self.neuron_weights)

        sorted_neuron_indices = np.argsort(distances)

        bmu_index = sorted_neuron_indices[0]
        second_bmu_index = sorted_neuron_indices[1]

        test_posture = input_vector.reshape(-1, 3)
        bmu_posture = self.neuron_weights[bmu_index].reshape(-1, 3)
        second_bmu_posture = self.neuron_weights[second_bmu_index].reshape(-1, 3)

        fig = plt.figure(figsize=(15, 5))

        ax1 = fig.add_subplot(131, projection="3d")
        self.plot_skeleton(
            ax1, test_posture, joint_connections, title=f"Test Sample {sample_index}"
        )

        ax2 = fig.add_subplot(132, projection="3d")
        self.plot_skeleton(
            ax2,
            bmu_posture,
            joint_connections,
            title=f"BMU {bmu_index}\nDistance={distances[bmu_index]:.4f}",
        )

        ax3 = fig.add_subplot(133, projection="3d")
        self.plot_skeleton(
            ax3,
            second_bmu_posture,
            joint_connections,
            title=f"2nd BMU {second_bmu_index}\nDistance={distances[second_bmu_index]:.4f}",
        )

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def plot_skeleton(self, ax, joints, joint_connections=None, title=""):
        """
        Plot 3D posture skeleton from joint coordinates.
        joints shape: (num_joints, 3)
        """

        x = joints[:, 0]
        y = joints[:, 1]
        z = joints[:, 2]

        ax.scatter(x, y, z, s=30)

        if joint_connections is not None:
            for joint_a, joint_b in joint_connections:
                ax.plot(
                    [x[joint_a], x[joint_b]],
                    [y[joint_a], y[joint_b]],
                    [z[joint_a], z[joint_b]],
                )

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    joint_connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (1, 4),
        (4, 5),
        (5, 6),
        # add based on your skeleton format
    ]

    # ------------------------------------------------------------
    # Supporting Methods
    # ------------------------------------------------------------
    def initialize_distance_normalization_constants(
        self, test_samples, neuron_counts=(range(6, 20, 2))
    ):
        """
        Compute normalization constants across multiple trained SOM models
        for discriminant and BMU-gap confidence scores. This function computes
        normalization constants from multiple trained SOMs so the discriminant
        and cluster confidence scores can be scaled consistently across models.
        """
        max_distances = []
        max_gap_distances = []

        for num_neurons in neuron_counts:
            model_file = f"SOM_{num_neurons}_cls.pkl"

            max_distances.append(self.get_max_bmu_distance(model_file, test_samples))

            max_gap_distances.append(
                self.get_max_gap_distance(model_file, test_samples)
            )

        self.sum_max_distances = sum(max_distances)
        self.sum_max_gap_distances = sum(max_gap_distances)

    def calculate_discriminant_score(self, neuron_distances, bmu_index):
        """
        Discriminant score based on BMU distance.
        Higher score = more confident assignment.
        """

        bmu_distance = neuron_distances[bmu_index]

        # If posture overlaps perfectly with BMU centroid
        if bmu_distance == 0:
            return 1.0

        inverse_distances = []

        for distance_value in neuron_distances:
            if distance_value == 0:
                inverse_distances.append(np.inf)
            else:
                inverse_distances.append(1 / abs(distance_value))

        # If any centroid has zero distance, it gets full confidence
        if np.isinf(inverse_distances).any():
            return 1.0 if bmu_distance == 0 else 0.0

        return inverse_distances[bmu_index] / np.sum(inverse_distances)

    def calculate_discriminant_gap_score(self, neuron_distances):
        """
        Confidence score based on BMU and 2nd BMU distance gap.
        Larger gap means more confident BMU assignment.
        """

        sorted_indices = np.argsort(neuron_distances)

        bmu_index = sorted_indices[0]
        second_bmu_index = sorted_indices[1]

        bmu_distance = neuron_distances[bmu_index]
        second_bmu_distance = neuron_distances[second_bmu_index]

        if second_bmu_distance == 0:
            return 1.0

        gap_score = (second_bmu_distance - bmu_distance) / second_bmu_distance

        return gap_score

    def get_max_bmu_distance(self, model_file, test_samples):
        """
        Maximum BMU distance over all test samples
        for a trained SOM model.
        """

        with open(model_file, "rb") as file_handle:
            model_data = pickle.load(file_handle)

        neuron_weights = model_data["neuron_weights"]

        max_bmu_distance = 0

        for posture_vector in test_samples:

            neuron_distances = self.calculate_distances(posture_vector, neuron_weights)

            bmu_distance = min(neuron_distances)

            max_bmu_distance = max(max_bmu_distance, bmu_distance)

        return max_bmu_distance

    def get_max_gap_distance(self, model_file, test_data):
        """Maximum BMU–2ndBMU gap per model."""
        with open(model_file, "rb") as f:
            model_dict = pickle.load(f)
        weights = model_dict["neuron_weights"]
        max_gap = 0
        for inp in test_data:
            distances = self.calculate_distances(inp, weights)
            sorted_d = np.sort(distances)
            gap = sorted_d[1] - sorted_d[0]
            max_gap = max(max_gap, gap)
        return max_gap

    def get_max_bmu_gap_distance(self, model_file, test_samples):
        """
        Maximum BMU-to-second-BMU distance gap over
        all test samples. This function finds the largest
        observed margin between first and second BMUs,
        useful for normalizing gap-based confidence scores.
        """
        with open(model_file, "rb") as file_handle:
            model_data = pickle.load(file_handle)

        neuron_weights = model_data["neuron_weights"]

        max_bmu_gap = 0

        for input_vector in test_samples:

            neuron_distances = self.calculate_distances(input_vector, neuron_weights)

            sorted_distances = np.sort(neuron_distances)

            if len(sorted_distances) < 2:
                continue

            bmu_distance = sorted_distances[0]
            second_bmu_distance = sorted_distances[1]

            bmu_gap = second_bmu_distance - bmu_distance

            max_bmu_gap = max(max_bmu_gap, bmu_gap)

        return max_bmu_gap

    # ------------------------------------------------------------
    # Visualization Helpers
    # ------------------------------------------------------------
    def plot_discriminant_series(
        self, bmu_sequence, discriminant_scores, cluster_colors, plot_title
    ):
        """
        Plot discriminant/confidence scores over the sequence,
        coloring each segment by BMU cluster.

        Also marks posture boundaries where the BMU changes.
        """

        plt.figure(figsize=(8, 6))
        labeled_clusters = set()

        for sample_index in range(len(discriminant_scores) - 1):

            cluster_id = bmu_sequence[sample_index]

            plt.plot(
                [sample_index, sample_index + 1],
                [
                    discriminant_scores[sample_index],
                    discriminant_scores[sample_index + 1],
                ],
                color=cluster_colors[cluster_id],
                linewidth=1.5,
            )

            if cluster_id not in labeled_clusters:
                plt.plot(
                    [],
                    [],
                    color=cluster_colors[cluster_id],
                    label=f"Cluster {cluster_id}",
                )
                labeled_clusters.add(cluster_id)

            # Mark posture boundary when BMU changes
            if bmu_sequence[sample_index] != bmu_sequence[sample_index + 1]:
                plt.axvline(
                    x=sample_index + 0.5, linestyle="--", linewidth=1, alpha=0.6
                )

        plt.xlabel("Sample Index")
        plt.ylabel("Discriminant Score")
        plt.title(plot_title)

        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

        plt.tight_layout()

        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/discriminant_series.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show(block=False)
        plt.pause(0.001)

    def plot_discriminant_subplots(self, bmu_sequence, input_samples, cluster_colors):
        """
        Plot discriminant score trends for each winning neuron.

        Each subplot corresponds to one BMU/neuron. Shaded regions show
        where that neuron is the active winner.
        """

        bmu_sequence = np.asarray(bmu_sequence)
        input_samples = np.asarray(input_samples)

        unique_bmus = np.unique(bmu_sequence)

        fig, axes = plt.subplots(
            len(unique_bmus), 1, figsize=(12, 4 * len(unique_bmus))
        )

        if len(unique_bmus) == 1:
            axes = [axes]

        fig.suptitle("Discriminant Scores for Each Winning Neuron")

        for subplot_index, neuron_index in enumerate(unique_bmus):

            neuron_discriminant_scores = []

            for input_vector in input_samples:
                neuron_distances = self.calculate_distances(
                    input_vector, self.neuron_weights
                )

                score = self.calculate_discriminant_score(
                    neuron_distances, neuron_index
                )

                neuron_discriminant_scores.append(score)

            axes[subplot_index].plot(
                neuron_discriminant_scores,
                color=cluster_colors[neuron_index],
                linewidth=1.5,
            )

            winning_sample_indices = np.where(bmu_sequence == neuron_index)[0]

            if len(winning_sample_indices) > 0:
                breaks = np.where(np.diff(winning_sample_indices) > 1)[0]

                segment_ranges = zip(
                    np.r_[0, breaks + 1], np.r_[breaks, len(winning_sample_indices) - 1]
                )

                for start_idx, end_idx in segment_ranges:
                    active_region = winning_sample_indices[start_idx : end_idx + 1]

                    axes[subplot_index].axvspan(
                        active_region[0],
                        active_region[-1],
                        color=cluster_colors[neuron_index],
                        alpha=0.2,
                    )

            axes[subplot_index].set_title(f"Neuron {neuron_index} Discriminant Scores")
            axes[subplot_index].set_xlabel("Sample Index")
            axes[subplot_index].set_ylabel("Discriminant Score")
            axes[subplot_index].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/discriminant_subplots.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.show(block=False)
        plt.pause(0.001)

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
        plt.title("SOM Weight Convergence Over Epochs")

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/weight_changes.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show(block=False)
        plt.pause(0.001)

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
        plt.title("Training Clusters (PCA Projection)")
        plt.grid(True, alpha=0.3)

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Clusters")

        plt.tight_layout()

        plt.savefig(
            f"figures_new_SOM_{self.num_neurons}/train_pca.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show(block=False)
        plt.pause(0.001)

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


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    training_data = "training_data"
    test_data = "test_data"

    test_samples = load_test_data(test_data)

    neuron_range = range(6, 20, 2)

    radius = 0.5
    learning_rate = 0.01
    gaussian = False
    dist_metric = "euclidean"

    num_epochs = 100
    convergence_threshold = 0.05
    patience = 10

    model_dir = "trained_som_models"
    os.makedirs(model_dir, exist_ok=True)

    results = []

    # ------------------------------------------------------------
    # Train SOM models from 5 to 20 neurons
    # ------------------------------------------------------------
    for num_neurons in neuron_range:
        print(f"\n================================================")
        print(f"Training SOM with {num_neurons} neurons")
        print(f"==================================================")

        os.makedirs(f"figures_new_SOM_{num_neurons}", exist_ok=True)

        som = CompetitiveLearning(
            num_neurons=num_neurons,
            training_data=training_data,
            radius=radius,
            learning_rate=learning_rate,
            gaussian=gaussian,
            dist_metric=dist_metric,
        )

        som.train(
            num_of_epochs=num_epochs,
            convergence_threshold=convergence_threshold,
            patience=patience,
        )

        model_path = f"{model_dir}/SOM_{num_neurons}_cls.pkl"
        som.save(model_path)

        print(f"Saved model: {model_path}")

        # --------------------------------------------------------
        # Load trained model
        # --------------------------------------------------------
        loaded_som = CompetitiveLearning(
            num_neurons=num_neurons,
            training_data=training_data,
            radius=radius,
            learning_rate=learning_rate,
            gaussian=gaussian,
            dist_metric=dist_metric,
        )

        loaded_som.load(model_path)

        # --------------------------------------------------------
        # Classify test data
        # --------------------------------------------------------
        bmu_sequence, second_bmu_sequence = loaded_som.classify(
            test_samples,
            min_run_length=10,
            apply_filter=True,
            plot_results=True,
            overlay_weights=True,
            max_points=6000,
        )

        # --------------------------------------------------------
        # Discriminant scores and gap scores
        # --------------------------------------------------------
        discriminant_scores = []
        gap_scores = []

        for input_vector in test_samples:
            neuron_distances = loaded_som.calculate_distances(
                input_vector, loaded_som.neuron_weights
            )

            sorted_indices = np.argsort(neuron_distances)
            bmu_index = sorted_indices[0]

            discriminant_score = loaded_som.calculate_discriminant_score(
                neuron_distances, bmu_index
            )

            gap_score = loaded_som.calculate_discriminant_gap_score(neuron_distances)

            discriminant_scores.append(discriminant_score)
            gap_scores.append(gap_score)

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
        )

        # --------------------------------------------------------
        # Plot posture match example
        # --------------------------------------------------------
        loaded_som.plot_posture_match(
            test_samples=test_samples,
            sample_index=0,
            joint_connections=loaded_som.joint_connections,
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
    plt.savefig("som_model_selection_metrics.png", dpi=200, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.001)

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
