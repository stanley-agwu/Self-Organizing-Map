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
from preprocess_all_act import load_training_data, load_test_data


# ================================================================
# Competitive Learning Class
# ================================================================
class CompetitiveLearning:
    def __init__(self, num_neurons, training_data, radius, learning_rate, gaussian, dist_metric):
        self.num_neurons = num_neurons
        self.radius = radius
        self.learning_rate = learning_rate
        self.gaussian = gaussian
        self.dist_metric = dist_metric

        self.input_data = load_training_data(training_data)
        self.neuron_weights = np.random.normal(
            np.mean(self.input_data),
            np.std(self.input_data),
            size=(self.num_neurons, len(self.input_data[0]))
        )

        # Containers
        self.potential = np.ones(self.num_neurons)
        self.activation = np.ones(self.num_neurons)
        self.winners_list = []
        self.error = []
        self.epoch_weights = []
        self.pca_results = []
        self.average_distance = []
        self.convergence_counter = 0


    # ------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------
    def calculate_distance(self, inp, for_calculate):
        """Calculate distance using the chosen metric."""
        metric_func = {
            'manhattan': distance.cityblock,
            'minkowski': lambda x, y: distance.minkowski(x, y, p=3),
            'hamming': distance.hamming,
            'cosine': distance.cosine,
            'euclidean': distance.euclidean
        }
        dist_func = metric_func.get(self.dist_metric, distance.euclidean)
        return [dist_func(neuron, inp) for neuron in for_calculate]


    def find_winner(self):
        """Find the neuron with minimum distance (BMU)."""
        avg_dist = np.average(self.distance)
        self.average_distance.append(avg_dist)
        neuron_idx = np.argmin(self.distance)
        return neuron_idx, np.min(self.distance)


    # ------------------------------------------------------------
    # Training procedure
    # ------------------------------------------------------------
    def update_weights(self, inp):
        """Update weights of the winner and its neighborhood."""
        self.neuron_weights[self.winner] += self.learning_rate * (inp - self.neuron_weights[self.winner])
        for idx, dist in self.kohonen_neighborhood().items():
            if dist <= self.radius:
                self.neuron_weights[idx] += 0.8 * self.learning_rate * (inp - self.neuron_weights[idx])

    def kohonen_neighborhood(self):
        """Return neurons within the neighborhood radius."""
        neighbors = {}
        for idx, weights in enumerate(self.neuron_weights):
            dist = self.calculate_distance(self.neuron_weights[self.winner], [weights])[0]
            if dist <= self.radius:
                neighbors[idx] = dist
        return neighbors


    def train(self, epoch_num, threshold=0.1, patience=10):
        """Train the SOM network."""
        self.all_steps = epoch_num * len(self.input_data)

        for epoch in range(epoch_num):
            print(f"Training Epoch {epoch}")
            self.input_data = shuffle(self.input_data)
            self.winners_list = []

            for inp in self.input_data:
                self.distance = self.calculate_distance(inp, self.neuron_weights)
                self.winner, _ = self.find_winner()
                self.winners_list.append(self.winner)
                self.update_weights(inp)

            self.calculate_error()

            if (epoch + 1) % 5 == 0:
                save_filename = f"SOM_neurons_{self.num_neurons}_epoch_{epoch + 1}.pkl"
                print(f"Saving model for {self.num_neurons} neurons at epoch {epoch + 1} → {save_filename}")
                self.save(save_filename)

            if self.check_convergence(epoch, threshold, patience):
                print("Early stopping due to convergence.")
                break

            self.epoch_weights.append(np.copy(self.neuron_weights))



        self.visualize_convergence()
        self.visualize_weight_pca()
        self.plot_weight_changes()
        self.plot_train()

        plt.figure(figsize=(8, 6))
        plt.plot(self.error, label="Training Error")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Training Error Over Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(f'figures_new_SOM_{self.num_neurons}/training_error.png')


    def calculate_error(self):
        """Quantization error."""
        errors = [min(self.calculate_distance(inp, self.neuron_weights)) ** 2 for inp in self.input_data]
        self.error.append(np.mean(errors))

    def get_cluster_colors(self, clusters):
        """Create a stable color map for cluster ids present in clusters."""
        unique = np.unique(clusters)
        cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
        return {c: cmap(i) for i, c in enumerate(unique)}

    def filter_short_runs(self, winners, X, min_len=10):
        """
        Keep only runs of consecutive equal winners with length >= min_len.
        Returns filtered_winners, filtered_X
        """
        winners = np.asarray(winners)
        X = np.asarray(X)

        if len(winners) == 0:
            return winners, X

        keep_idx = []
        run_start = 0
        for i in range(1, len(winners)):
            if winners[i] != winners[i - 1]:
                run_len = i - run_start
                if run_len >= min_len:
                    keep_idx.extend(range(run_start, i))
                run_start = i

        # last run
        run_len = len(winners) - run_start
        if run_len >= min_len:
            keep_idx.extend(range(run_start, len(winners)))

        keep_idx = np.asarray(keep_idx, dtype=int)
        return winners[keep_idx], X[keep_idx]

    def plot_winning_clusters_2d(self, X, winners, title,
                                save_path=None, max_points=None,
                                overlay_weights=True, learning_rate=0.7, s=14):
        X = np.asarray(X)
        winners = np.asarray(winners)

        if len(X) == 0:
            print("Nothing to plot (empty X).")
            return

        # optionally subsample
        if max_points is not None and len(X) > max_points:
            idx = np.random.choice(len(X), size=max_points, replace=False)
            Xp = X[idx]
            wp = winners[idx]
        else:
            Xp = X
            wp = winners

        # PCA projection (data + weights)
        if overlay_weights:
            pca = PCA(n_components=2)
            Z = np.vstack([Xp, self.neuron_weights])
            Z2 = pca.fit_transform(Z)
            X2 = Z2[:len(Xp)]
            W2 = Z2[len(Xp):]
        else:
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(Xp)
            W2 = None

        colors = self.get_cluster_colors(wp)

        # ✅ THIS MUST BE INSIDE THE FUNCTION
        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(right=0.75)

        for c in np.unique(wp):
            idx = np.where(wp == c)[0]
            plt.scatter(X2[idx, 0], X2[idx, 1],
                        color=colors[c], label=f"Neuron {c}",
                        learning_rate=learning_rate, s=s, edgecolors='none')

        if overlay_weights and W2 is not None:
            plt.scatter(W2[:, 0], W2[:, 1],
                        marker='x', s=80, linewidths=1.5,
                        color='black', label='Neuron weights')

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(title)
        plt.grid(True, learning_rate=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Winners")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200)


    # ------------------------------------------------------------
    # Convergence visualization
    # ------------------------------------------------------------
    def check_convergence(self, epoch, threshold, patience):
        """Stop if PCA changes fall below a threshold for multiple epochs."""
        if len(self.epoch_weights) > 1:
            pca = PCA(n_components=2)
            weights_matrix = np.array(self.epoch_weights).reshape((len(self.epoch_weights), -1))
            pca_result = pca.fit_transform(weights_matrix)
            self.pca_results.append(pca_result[-1])

            if len(self.pca_results) > 1:
                pca_change = np.linalg.norm(self.pca_results[-2] - self.pca_results[-1])
                if pca_change < threshold:
                    self.convergence_counter += 1
                    if self.convergence_counter >= patience:
                        return True
                else:
                    self.convergence_counter = 0
        return False


    def visualize_convergence(self):
        """Visualize PCA evolution over epochs."""
        if len(self.pca_results) < 2:
            return
        pca_array = np.array(self.pca_results)
        epochs = range(1, len(pca_array) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, pca_array[:, 0], label='PCA Comp 1')
        plt.plot(epochs, pca_array[:, 1], label='PCA Comp 2')
        plt.xlabel("Epoch")
        plt.ylabel("PCA Value")
        plt.title("Convergence of Weights (PCA Space)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f'figures_new_SOM_{self.num_neurons}/convergence_pca.png')


    def visualize_weight_pca(self):
        """Visualize PCA trajectory of weight updates."""
        weights_matrix = np.array(self.epoch_weights).reshape((len(self.epoch_weights), -1))
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(weights_matrix)

        plt.figure(figsize=(8, 6))
        plt.plot(pca_result[:, 0], pca_result[:, 1], marker='o')
        plt.xlabel("PCA Comp 1")
        plt.ylabel("PCA Comp 2")
        plt.title("PCA of Weight Evolution")
        plt.grid(True)

        plt.savefig(f'figures_new_SOM_{self.num_neurons}/weight_pca.png')


    # ------------------------------------------------------------
    # Classification (Discriminant Score Evaluation)
    # ------------------------------------------------------------
    def classify(self, X_test, min_run_len=10, do_filter=True,
             plot=True, overlay_weights=True, max_points=6000):
        """
        Classify test data using BMU only.
        Optionally filter short non-consecutive runs (<min_run_len).
        Optionally plot PCA scatter colored by winning neuron.
        """
        X_test = np.asarray(X_test)

        winners = []
        second_winners = []

        for inp in X_test:
            disto = self.calculate_distance(inp, self.neuron_weights)
            sorted_idx = np.argsort(disto)
            winners.append(sorted_idx[0])
            second_winners.append(sorted_idx[1])

        winners = np.asarray(winners)
        second_winners = np.asarray(second_winners)

        X_plot = X_test
        winners_plot = winners

        if do_filter:
            winners_plot, X_plot = self.filter_short_runs(winners, X_test, min_len=min_run_len)
            if len(X_plot) == 0:
                print("No clusters survived filtering.")
                # still return raw results
                return winners, second_winners

        if plot:
            self.plot_winning_clusters_2d(
                X_plot, winners_plot,
                title=f"Winning Neurons (BMU) on Test Data (PCA 2D) | min_run_len={min_run_len if do_filter else 0}",
                save_path=f'figures_new_SOM_{self.num_neurons}/winning_clusters_bmu_pca.png',
                max_points=max_points,
                overlay_weights=overlay_weights
            )
            print(f'figures_new_SOM_{self.num_neurons}/winning_clusters_bmu_pca.png, plotted')

        return winners, second_winners

    # ------------------------------------------------------------
    # Supporting Methods
    # ------------------------------------------------------------
    def initialize_max_distances(self, test_data):
        """Compute normalization constants across 3 trained models."""
        self.max_dist_10 = self.get_max_distance(f'SOM_{self.num_neurons}_cls.pkl', test_data)
        self.max_dist_12 = self.get_max_distance(f'SOM_{self.num_neurons}_cls.pkl', test_data)
        self.max_dist_14 = self.get_max_distance(f'SOM_{self.num_neurons}_cls.pkl', test_data)
        self.sum_max_distances = self.max_dist_10 + self.max_dist_12 + self.max_dist_14

        self.max_gap_10 = self.get_max_gap_distance(f'SOM_{self.num_neurons}_cls.pkl', test_data)
        self.max_gap_12 = self.get_max_gap_distance(f'SOM_{self.num_neurons}_cls.pkl', test_data)
        self.max_gap_14 = self.get_max_gap_distance(f'SOM_{self.num_neurons}_cls.pkl', test_data)
        self.sum_max_gap = self.max_gap_10 + self.max_gap_12 + self.max_gap_14

    def calculate_discriminant(self, disto, winner):
        """Discriminant score based on BMU distance."""
        return 1 - (disto[winner] / self.sum_max_distances)

    def calculate_discriminant_gap(self, disto):
        """Discriminant based on BMU–2ndBMU gap."""
        sorted_d = np.sort(disto)
        gap = sorted_d[1] - sorted_d[0]
        return 1 - (gap / self.sum_max_gap)


    def get_max_distance(self, model_file, test_data):
        """Maximum BMU distance per model."""
        with open(model_file, 'rb') as f:
            model_dict = pickle.load(f)
        weights = model_dict['neuron_weights']
        max_dist = 0
        for inp in test_data:
            distances = self.calculate_distance(inp, weights)
            max_dist = max(max_dist, max(distances))
        return max_dist

    def get_max_gap_distance(self, model_file, test_data):
        """Maximum BMU–2ndBMU gap per model."""
        with open(model_file, 'rb') as f:
            model_dict = pickle.load(f)
        weights = model_dict['neuron_weights']
        max_gap = 0
        for inp in test_data:
            distances = self.calculate_distance(inp, weights)
            sorted_d = np.sort(distances)
            gap = sorted_d[1] - sorted_d[0]
            max_gap = max(max_gap, gap)
        return max_gap


    # ------------------------------------------------------------
    # Visualization Helpers
    # ------------------------------------------------------------
    def plot_discriminant_series(self, clusters, scores, colors, title):
        plt.figure(figsize=(8, 6))
        labeled = set()
        for i in range(len(scores) - 1):
            c = clusters[i]
            plt.plot([i, i + 1], [scores[i], scores[i + 1]],
                     color=colors[c % len(colors)])
            if c not in labeled:
                plt.plot([], [], color=colors[c % len(colors)], label=f'Cluster {c}')
                labeled.add(c)
        plt.xlabel("Data Points")
        plt.ylabel("Discriminant Score")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'figures_new_SOM_{self.num_neurons}/discriminant_series.png')

    def plot_discriminant_subplots(self, clusters, X_data, colors):
        """Subplots showing discriminant trends per winning neuron."""
        unique_winners = np.unique(clusters)
        fig, axs = plt.subplots(len(unique_winners), 1, figsize=(12, 4 * len(unique_winners)))
        if len(unique_winners) == 1:
            axs = [axs]
        fig.suptitle('Discriminant Scores for Each Winning Neuron')

        for idx, neuron in enumerate(unique_winners):
            neuron_scores = []
            for x in X_data:
                disto = self.calculate_distance(x, self.neuron_weights)
                score = 1 - (disto[neuron] / self.sum_max_distances)
                neuron_scores.append(score)

            axs[idx].plot(neuron_scores, color=colors[neuron % len(colors)])
            winner_idx = np.where(np.array(clusters) == neuron)[0]

            if len(winner_idx) > 0:
                from numpy import diff, where, r_
                breaks = where(diff(winner_idx) > 1)[0]
                ranges = zip(r_[0, breaks + 1], r_[breaks, len(winner_idx)])
                for start, end in ranges:
                    region = winner_idx[start:end + 1]
                    axs[idx].axvspan(region[0], region[-1],
                                     color=colors[neuron % len(colors)], learning_rate=0.2)

            axs[idx].set_title(f'Neuron {neuron} Discriminant Scores')
            axs[idx].set_xlabel('Data Points')
            axs[idx].set_ylabel('Discriminant')
            axs[idx].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'figures_new_SOM_{self.num_neurons}/discriminant_subplots.png')


    # ------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------
    def plot_weight_changes(self):
        """Plot norm difference between consecutive epoch weights."""
        diffs = [np.linalg.norm(self.epoch_weights[i + 1] - self.epoch_weights[i])
                 for i in range(len(self.epoch_weights) - 1)]
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(diffs) + 1), diffs, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Weight Δ (L2 norm)")
        plt.title("Weight Evolution Over Epochs")
        plt.grid(True)

        plt.savefig(f'figures_new_SOM_{self.num_neurons}/weight_changes.png')

    def plot_train(self):
        """Plot PCA visualization of trained clusters."""
        pca = PCA(n_components=2)
        reduced_input = pca.fit_transform(self.input_data)
        plt.figure(figsize=(10, 8))
        colors = [plt.cm.Set1(i / self.num_neurons) for i in range(self.num_neurons)]
        for i, inp in enumerate(reduced_input[:4000]):
            plt.scatter(inp[0], inp[1], color=colors[self.winners_list[i]])
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=colors[i], markersize=10,
                                  label=f'Cluster {i + 1}') for i in range(self.num_neurons)]
        plt.legend(handles=legend_elements, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Training Clusters (PCA Reduced)")
        plt.savefig(f'figures_new_SOM_{self.num_neurons}/train_pca.png')


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))



#================================================================
# MAIN EXECUTION
# ================================================================
# if __name__ == '__main__':
#     param = "param"
#     for num_neurons in [10, 12, 14]:
#         os.makedirs(f'figures_new_SOM_{num_neurons}', exist_ok=True)

#         CL = CompetitiveLearning(
#             num_neurons=num_neurons,
#             training_data="param",
#             radius=0.0001,
#             learning_rate=0.007,
#             gaussian=1,
#             dist_metric='cosine'
#         )

#         CL.train(epoch_num=40)
#         CL.save(f'SOM_{num_neurons}_cls.pkl')

#         test_data = load_test_data('test')

#         winners, second_winners = CL.classify(
#             test_data,
#             min_run_len=10,
#             do_filter=True,
#             plot=True,
#             overlay_weights=True,
#             max_points=8000
#         )

#         colors = [plt.cm.Set1(i / CL.num_neurons) for i in range(CL.num_neurons)]
#         CL.plot_discriminant_subplots(winners, test_data, colors)


#         print(f"winners for {num_neurons} neurons:", winners[:100])
#         print(f"second_winners for {num_neurons} neurons:", second_winners[:100])

#         plt.show(block=False)
#         plt.pause(1)
#         plt.close('all')
    
#     # initialize normalization constants first
#     CL.initialize_max_distances(test_data)

#     scores = []
#     for x in test_data:
#         disto = CL.calculate_distance(x, CL.neuron_weights)
#         winner = np.argmin(disto)
#         score = 1 - (disto[winner] / CL.sum_max_distances)
#         scores.append(score)
#     CL.plot_discriminant_series(
#     winners,
#     scores,
#     colors,
#     title=f"Discriminant Series ({num_neurons} neurons)"
# )

if __name__ == '__main__':
    param = "param"
    number_of_neurons_list = [10, 12, 14]

    # PASS 1: train and save all models
    for num_neurons in number_of_neurons_list:
        os.makedirs(f'figures_new_SOM_{num_neurons}', exist_ok=True)

        CL = CompetitiveLearning(
            num_neurons=num_neurons,
            training_data=param,
            radius=0.0001,
            learning_rate=0.007,
            gaussian=1,
            dist_metric='cosine'
        )

        CL.train(epoch_num=40)
        CL.save(f'SOM_{num_neurons}_cls.pkl')

    # Load test data once
    test_data = load_test_data('test')

    # PASS 2: load each trained model and make discriminant plots
    for num_neurons in number_of_neurons_list:
        CL = CompetitiveLearning(
            num_neurons=num_neurons,
            training_data="training_data",
            radius=0.0001,
            learning_rate=0.007,
            gaussian=1,
            dist_metric='cosine'
        )

        CL.load(f'SOM_{num_neurons}_cls.pkl')

        # initialize normalization constants
        CL.initialize_max_distances(test_data)

        winners, second_winners = CL.classify(
            test_data,
            min_run_len=10,
            do_filter=True,
            plot=True,
            overlay_weights=True,
            max_points=8000
        )

        scores = []
        for x in test_data:
            disto = CL.calculate_distance(x, CL.neuron_weights)
            winner = np.argmin(disto)
            score = 1 - (disto[winner] / CL.sum_max_distances)
            scores.append(score)

        colors = [plt.cm.Set1(i / CL.num_neurons) for i in range(CL.num_neurons)]

        CL.plot_discriminant_subplots(winners, test_data, colors)

        CL.plot_discriminant_series(
            winners,
            scores,
            colors,
            title=f"Discriminant Series ({num_neurons} neurons)"
        )

        print(f"winners for {num_neurons} neurons:", winners[:100])
        print(f"second_winners for {num_neurons} neurons:", second_winners[:100])

        plt.show(block=False)
        plt.pause(1)
        plt.close('all')
