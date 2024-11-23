import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import MinMaxNorm, NonNeg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import io

# Création de la couche personnalisée "LineGain"
class LineGainLayer(layers.Layer):
    def __init__(self, units, line_gain, **kwargs):
        super(LineGainLayer, self).__init__(**kwargs)
        self.units = units
        self.line_gain = line_gain

    def build(self, input_shape):
        # Initialise un vecteur de poids multiplicatifs avec ou sans régularisation L1
        self.gains = self.add_weight(
            shape=(input_shape[-1],),
            initializer= tf.keras.initializers.Constant(1),
            regularizer=regularizers.L1(self.line_gain),  # Régularisation L1 si activée
            trainable=True,
            constraint=NonNeg(),
            name="gains"
        )
    def call(self, inputs):
        return inputs * self.gains

    def get_gains(self):
        return self.gains.numpy()  


# Création du callback personnalisé pour Early Stopping basé sur LineGainLayer
class EarlyStoppingByLineGain(tf.keras.callbacks.Callback):
    def __init__(self, layer, epsilon, N):
        super(EarlyStoppingByLineGain, self).__init__()
        self.layer = layer
        self.epsilon = epsilon
        self.N = N
        self.lowest_gain_index = None

    def on_epoch_end(self, epoch, logs=None):
        gains = self.layer.get_gains()
        self.lowest_gain_index = np.argmin(np.abs(gains))

        print (max(np.abs(gains)) - min(np.abs(gains)) - np.mean(np.abs(gains)))
        if epoch % self.N == 0:
            loss = logs.get('loss')
            accuracy = logs.get('accuracy')
            print(f"Epoch {epoch + 1} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Gains: {gains}")

        if gains[self.lowest_gain_index] < self.epsilon:
            print(f"\nArrêt de l'apprentissage à l'époque {epoch + 1}")
            print(f"Indice de l'entrée à éliminer : {self.lowest_gain_index}")
            print(f"Valeur des gains : {gains}")
            self.model.stop_training = True


# Callback pour surveiller la stabilité de la perte
class LossStabilityCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, min_delta=0.001):
        super(LossStabilityCallback, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = np.Inf
        self.stable = False

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Stabilité atteinte après {epoch + 1} epochs")
                self.stable = True
                self.model.stop_training = True

# Fonction pour construire le modèle avec ou sans régularisation L1
def model_Test(input_shape, line_gain, gains=None):
    if (line_gain == 0):
        line_gain_layer = None
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(5, activation="tanh"),
            layers.Dense(5, activation="tanh"),
            layers.Dense(4, activation="tanh"),
            layers.Dense(2, activation="tanh"),
            layers.Dense(1, activation="sigmoid")
        ])
    else :
        line_gain_layer = LineGainLayer(units=input_shape, line_gain=line_gain)
        L2_factor = 0.001
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_shape),
            line_gain_layer,
            layers.Dense(5, activation="tanh", kernel_regularizer=regularizers.L2(L2_factor)),
            layers.Dense(5, activation="tanh", kernel_regularizer=regularizers.L2(L2_factor)),
            layers.Dense(4, activation="tanh", kernel_regularizer=regularizers.L2(L2_factor)),
            layers.Dense(2, activation="tanh", kernel_regularizer=regularizers.L2(L2_factor)),
            layers.Dense(1, activation="sigmoid")
        ])
        # if gains is not None:
        #     line_gain_layer.gains.assign(gains)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, line_gain_layer

def generate_spiral_data(n_points, noise=0.2, factor = 10):
    n = np.sqrt(np.random.rand(int(n_points/2),1)) * factor 
    d1x =(-np.cos(n)*n + np.random.rand(int(n_points/2),1) * noise) / factor
    d1y =(np.sin(n)*n + np.random.rand(int(n_points/2),1) * noise) / factor 
    X = np.vstack((np.hstack((d1x,d1y)), np.hstack((-d1x,-d1y))))
    y = np.hstack((np.zeros(int(n_points/2)), np.ones(int(n_points/2))))
    return X, y

# Circle dataset
def generate_circle_data(n_points, noise=0.1):
    n_points = int(n_points/2)
    radius = np.sqrt(np.random.rand(n_points))
    angle = 2 * np.pi * np.random.rand(n_points)
    X_inner = np.vstack([radius * np.cos(angle), radius * np.sin(angle)]).T
    X_outer = np.vstack([(radius + 1.7) * np.cos(angle), (radius + 1.7) * np.sin(angle)]).T
    X = (np.vstack([X_inner, X_outer]) + noise * np.random.randn(2 * n_points, 2))/3
    y = np.hstack([np.zeros(n_points), np.ones(n_points)])
    return X, y

# XOR dataset
def generate_xor_data(n_points, separator=0.02):
    n_points = n_points // 4
    # Cluster for Class 0
    X1 = -np.random.rand(n_points, 2) - separator
    X2 = np.random.rand(n_points, 2) + separator
    # Cluster for Class 1
    X3 = np.hstack([-np.random.rand(n_points, 1)- separator , np.random.rand(n_points, 1)+ separator])
    X4 = np.hstack([np.random.rand(n_points, 1)+ separator , -np.random.rand(n_points, 1)- separator])
    
    X = np.vstack([X1, X2, X3, X4])
    y = np.hstack([np.zeros(2 * n_points), np.ones(2 * n_points)])
    return X, y

def generate_gaussian_data(n_points, noise=0.05):
    # Divide the points between the two clusters
    n_points = int(n_points / 2)
    
    # Define tighter cluster centers within the range [-0.5, 0.5]
    mean1 = [0.5, 0.5]
    mean2 = [-0.5, -0.5]
    
    # Smaller covariance matrix for sparser clusters
    cov = [[noise, 0], [0, noise]]
    
    # Generate two clusters with the defined means and covariance
    X1 = np.random.multivariate_normal(mean1, cov, n_points)
    X2 = np.random.multivariate_normal(mean2, cov, n_points)
    
    # Combine clusters and add minimal noise, keeping points within [-1, 1]
    X = np.vstack([X1, X2])
    
    # Labels: 0 for the first cluster, 1 for the second
    y = np.hstack([np.zeros(n_points), np.ones(n_points)])
    
    return X, y

def update(epoch, model, ax, grid, xx, yy, update_interval, datas_shuffled, labels_shuffled):
    # Entraînement pour l'intervalle donné
    model.fit(datas_shuffled, labels_shuffled, epochs=update_interval, verbose=1)

    # Supprimez le contour précédent s'il existe
    if hasattr(update, "contour") and update.contour is not None:
        for c in update.contour.collections:
            c.remove() # sinon supperposition , on voit plus rien 

    # Prédiction sur la grille pour visualiser la frontière de décision
    Z = model.predict(grid).reshape(xx.shape)
    update.contour = ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.5)
    ax.set_title(f"Frontière de décision - Epoch: {(epoch+1) * update_interval}")

def plot_decision_boundary(ax, model, grid, xx, yy, X, y, feature_name=None, removed_features=None, accuracy=None):
    """Plots the decision boundary and displays removed features in the title along with accuracy."""
    ax.clear()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", alpha=0.7)
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    removed_text = f"Removed features: {', '.join(removed_features)}" if removed_features else "No features removed yet"
    title = f"{removed_text}" if feature_name else "Initialisation" # \nCurrent feature removed: {feature_name}
    title += f"\nAccuracy: {accuracy:.2%}" if accuracy is not None else ""
    ax.set_title(title)

def save_snapshot(fig):
    """Save a snapshot of the current figure as an RGB array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi)
    buf.seek(0)
    img = plt.imread(buf)
    buf.close()
    return img

def Learning(dataset_point, labels, epoch, line_gain=0.001, epsilon=0.001, x=True, y=True, x_2=True, y_2=True, xy=True, sin_x=True, sin_y=True, show_graph=False, update_interval=10):
    """Main function to manage feature selection, model training, and feature pruning visualization."""
    # Build initial features and their names
    features = []
    if x: features.append(dataset_point[:, 0])
    if y: features.append(dataset_point[:, 1])
    if x_2: features.append(dataset_point[:, 0] ** 2)
    if y_2: features.append(dataset_point[:, 1] ** 2)
    if xy: features.append(dataset_point[:, 0] * dataset_point[:, 1])
    if sin_x: features.append(np.sin(dataset_point[:, 0]))
    if sin_y: features.append(np.sin(dataset_point[:, 1]))
    
    feature_names = []
    if x: feature_names.append("x")
    if y: feature_names.append("y")
    if x_2: feature_names.append("x^2")
    if y_2: feature_names.append("y^2")
    if xy: feature_names.append("xy")
    if sin_x: feature_names.append("sin(x)")
    if sin_y: feature_names.append("sin(y)")

    # Record of deleted features for display purposes
    removed_features = []

    # Prepare dataset for training
    datas = np.column_stack(features)
    shuffled_indices = np.random.permutation(len(datas))
    datas_shuffled = datas[shuffled_indices]
    labels_shuffled = labels[shuffled_indices]

    # Initial model setup
    nb_input = datas_shuffled.shape[1]
    model, _ = model_Test(nb_input, line_gain=0)  # no line gain for pre-training

    # Callback to monitor loss stability
    loss_stability_callback = LossStabilityCallback(patience=5, min_delta=0.001)
    model.fit(datas_shuffled, labels_shuffled, epochs=epoch, verbose=1, callbacks=[loss_stability_callback])
    accuracy = model.evaluate(datas_shuffled, labels_shuffled, verbose=0)[1]  # Initial accuracy
    
    # Plot initial decision boundary
    snapshots = []  # List to save each intermediate plot
    if show_graph:
        # Set up grid for decision boundary plotting
        xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        base_grid = np.c_[xx.ravel(), yy.ravel()]
        grid_features = []
        if x: grid_features.append(base_grid[:, 0])
        if y: grid_features.append(base_grid[:, 1])
        if x_2: grid_features.append(base_grid[:, 0] ** 2)
        if y_2: grid_features.append(base_grid[:, 1] ** 2)
        if xy: grid_features.append(base_grid[:, 0] * base_grid[:, 1])
        if sin_x: grid_features.append(np.sin(base_grid[:, 0]))
        if sin_y: grid_features.append(np.sin(base_grid[:, 1]))
        grid = np.column_stack(grid_features)
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100, constrained_layout=False)
        plot_decision_boundary(ax, model, grid, xx, yy, dataset_point, labels, accuracy=accuracy)
        snapshots.append(save_snapshot(fig))  
        plt.pause(1)
    # If stability is reached, enable LineGain and start feature pruning
    if loss_stability_callback.stable:

        stable_weights = [layer.get_weights() for layer in model.layers if len(layer.get_weights()) > 0]
        model, line_gain_layer = model_Test(nb_input, line_gain)
        # Transfer weights to the new model with LineGain
        for i, layer_weights in enumerate(stable_weights):
            model.layers[i + 1].set_weights(layer_weights)

        early_stopping_callback = EarlyStoppingByLineGain(layer=line_gain_layer, epsilon=epsilon, N=update_interval)

        while True:
            # Train model with callback for early stopping by LineGain
            model.fit(datas_shuffled, labels_shuffled, epochs=epoch, verbose=0, callbacks=[early_stopping_callback])
            accuracy = model.evaluate(datas_shuffled, labels_shuffled, verbose=0)[1]

            if early_stopping_callback.model.stop_training:
                # Identify the feature to be removed
                lowest_gain_index = early_stopping_callback.lowest_gain_index
                feature_name = feature_names.pop(lowest_gain_index)
                removed_features.append(feature_name)  # Add removed feature to the record
                print(f"Deleted feature: {feature_name}")
                # Display updated decision boundary
                if show_graph:
                    # Update grid features to match the reduced number of inputs
                    plot_decision_boundary(ax, model, grid, xx, yy, dataset_point, labels, feature_name=feature_name, removed_features=removed_features, accuracy=accuracy)
                    snapshots.append(save_snapshot(fig))  
                    plt.pause(1)
                    grid_features.pop(lowest_gain_index) # Delete input in grid
                    grid = np.column_stack(grid_features)

                # Delete input in data
                nb_input -= 1
                datas_shuffled = np.delete(datas_shuffled, lowest_gain_index, axis=1)
                previous_gains = np.delete(line_gain_layer.get_gains(), lowest_gain_index)
                # Update model to reflect the reduced input size
                new_model, new_line_gain_layer = model_Test(nb_input, line_gain, gains=previous_gains)
                for i in range(2, len(model.layers)):  # Ignore InputLayer and LineGainLayer
                    new_model.layers[i].set_weights(model.layers[i].get_weights())

                model, line_gain_layer = new_model, new_line_gain_layer
                early_stopping_callback = EarlyStoppingByLineGain(layer=new_line_gain_layer, epsilon=epsilon, N=update_interval)
            else:
                print("No more features to delete. Training complete.")
                if show_graph:
                    # Final accuracy and plot
                    plot_decision_boundary(ax, model, grid, xx, yy, dataset_point, labels, feature_name="Final Model", removed_features=removed_features, accuracy=accuracy)
                    snapshots.pop(-1)
                    snapshots.append(save_snapshot(fig))  
                    plt.pause(1)
                break

        if show_graph :
            # Display all snapshots in a multi-plot grid
            num_snapshots = len(snapshots)
            cols = 3
            rows = math.ceil(num_snapshots / cols)
            fig_gallery, axes = plt.subplots(rows, cols, figsize=(15, 7 * rows))
            axes = axes.flatten()

            for i, snapshot in enumerate(snapshots):
                ax_img = axes[i]  # Use a different variable name
                ax_img.imshow(snapshot)
                ax_img.axis('off')

            # Hide any extra subplots that don't have snapshots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.show()
    else:
        print("Loss did not stabilize within the pre-training period.")

np.set_printoptions(linewidth=np.inf) # Configuration pour afficher les arrays numpy sur une seule ligne
#dataset_point, labels = generate_spiral_data(1000)
#dataset_point, labels = generate_gaussian_data(1000)
dataset_point, labels = generate_circle_data(1000)
#dataset_point, labels = generate_xor_data(1000)
#print("Shape: ", dataset_point.shape)

#Visualisation du dataset
# plt.figure(figsize=(8, 6))
# plt.scatter(dataset_point[:, 0], dataset_point[:, 1], c=labels, cmap="bwr", edgecolor="k", s=50, alpha=0.7)
# plt.title("Dataset Spirale - Deux Classes")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

Learning(dataset_point, labels, epoch = 500, line_gain= 0.007, show_graph = True)



