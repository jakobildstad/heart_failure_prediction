from pathlib import Path #OS-independent file paths
import pickle # python object serialization for saving model metadata
import torch # PyTorch for model training and evaluation
from torch.utils.data import DataLoader 
# DataLoader is a mini-batch iterator for that can shuffle and parallell load data.

from modules.preprocess import (
    # Everything that touches raw csv data and turns into tensors. 
    load_data, # Load the raw CSV data into a pandas DataFrame.
    prepare_features, # Prepare the features by scaling and splitting the data into training, validation, and test sets.
    TabularDataset, # A custom Dataset class that wraps the feature and target tensors for tabular data.
    save_scaler, # Save the fitted StandardScaler to a file for later use in production.
)

# HeartNet is the model architecture, meaning the neural network. 
# it is implemented as a subclass of `torch.nn.Module`.
# It defines the forward pass and the model's structure.
# The model is trained using the `train` function, which handles the training loop,
# loss calculation, and optimization of the model's parameters (weights).
# The `train` function also includes validation to monitor the model's performance
# on a validation set during training, which helps in preventing overfitting.
# The model is trained on a training set and validated on a validation set and tested on a testing set. 
# The `evaluate` function is used to assess the model's performance on a test set.
# These functions are defined in the `modules.trainer` module.
from modules.trainer import HeartNet, train, evaluate

# ────────────────────────────────────────────────────────────────
# Config  (change here, not in library code)
# ────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/heart.csv") # Path to the raw CSV data file
BATCH_SIZE = 64 #how many rows the model sees per gradient update.
EPOCHS = 200 # maximum number of times the model will see the entire dataset.
# The model will stop training early if the validation loss does not improve for a certain number of epochs (patience).
PATIENCE = 10
# Here patience is set to 10, meaning if the validation loss does not improve for 10 consecutive epochs,
# the training will stop early.
# This helps in saving time and resources by not allowing the model to train unnecessarily
# when it has already reached its optimal performance on the validation set.
LEARNING_RATE = 1e-3 # Step size for Adam, which is the optimizer used for training.
# Adam is an adaptive learning rate optimization algorithm that adjusts the learning rate for each parameter individually.'
# It is widely used in deep learning due to its efficiency and effectiveness.
# The learning rate determines how much to change the model's parameters in response to the estimated error
# each time the model weights are updated.
# A smaller learning rate means the model learns more slowly, while a larger learning rate means it
# learns more quickly but may overshoot the optimal solution.
# The learning rate is a crucial hyperparameter that can significantly affect the training process and the final
# performance of the model.
# The learning rate is set to 0.001 (1e-3), which is a common starting point for many deep learning tasks.
# It can be adjusted based on the specific problem and the behavior of the model during training.
# If the model is not converging or is oscillating, the learning rate may need to be decreased.
# If the model is converging too slowly, the learning rate may be increased.
# It is often beneficial to experiment with different learning rates to find the best one for a specific task.
# The learning rate can also be adjusted dynamically during training
# using learning rate schedulers, which can help improve convergence and final performance.
DEVICE = "cpu"  # "cpu" or "cuda" for GPU acceleration if that is available. 


def training_pipeline():
    # 1. Load + preprocess
    print("Loading data …")
    df = load_data(DATA_PATH) #pd.read_csv, makes a DataFrame (df) )bject from the CSV file.

    (
        X_train, # X is the feature matric/vector that the model gets as input.
        y_train, # y is the target vector that the model tries to predict.
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        feature_names,
    ) = prepare_features(df) #<---------- Check this out
    # Turns the DataFrame into PyTorch tensors for training, validation, and testing.
    # The `prepare_features` function also scales the numerical features using a `StandardScaler`,
    # which standardizes the features by removing the mean and scaling to unit variance. 
    # This is important for many machine learning algorithms, including neural networks,
    # as it helps to ensure that all features contribute equally to the model's learning process.

    # Save scaler for production use.
    save_scaler(scaler, "model/scaler.pkl")

    # 2. Build DataLoaders
    train_loader = DataLoader(
        # DataLoader is a PyTorch utility that provides an iterable over the dataset.
        # It allows you to load data in batches, shuffle the data, and use multiple workers
        # for loading data in parallel, which can speed up the training process.
        # TabularDataset is a small wrapper that allows DataLoader to batch the rows.
        # Training data is shuffled to break any potential order in the data,
        # which helps the model generalize better.
        TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TabularDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )
    # DataLoader yields batches of data for training, validation, and testing, 
    # which then get iterated over during training and evaluation.

    # 3. Instantiate model.
    # HeartNet is a custom neural network model defined in the `modules.trainer` module.
    # It is a subclass of `torch.nn.Module` and defines the architecture of the model
    # including the layers and the forward pass.
    # The model is initialized with the number of features in the input data.
    model = HeartNet(num_features=X_train.shape[1]) #<---------- Check this out
    print(model)

    # 4. Train
    # Trains the model using the training data and validates it using the validation data.
    # The `train` function handles the training loop, loss calculation, and optimization of the
    # model's parameters (weights). It also includes early stopping to prevent overfitting.
    # The `train` function returns the trained model with the best weights saved.
    model = train( #<---------- Check this out
        model, # The HeartNet model instance.
        train_loader, # DataLoader for training data.
        val_loader, # DataLoader for validation data.
        epochs=EPOCHS, # Number of epochs to train the model.
        lr=LEARNING_RATE, # Learning rate for the optimizer.
        device=DEVICE, # Device to run the training on (CPU or GPU).
        patience=PATIENCE, # Number of epochs to wait for improvement before stopping training.
        ckpt_path="model/best_heartnet.pth", # Path to save the best model weights.
    )
    # train() runds epock loops: forward pass, loss calculation, backward pass, and optimizer step.
    ### 1. forward poss: The model processes the input data and produces predictions.
    ### 2. loss calculation: The model's predictions are compared to the true labels using a loss function
    ### 3. backward pass: The gradients of the loss with respect to the model's parameters are computed.
    ### 4.  optimizer step: The optimizer updates the model's parameters based on the computed gradients.
    # It does this loop until the model has seen the entire training dataset `EPOCHS` times or early stopping is triggered.
    #  - tracks validation loss each epoch, and if it hasnt improved for `patience` epochs, it stops training early.
    # Whenever val-loss is best-seen, weights are saved to `ckpt_path`.

    # 5. Evaluate
    metrics = evaluate(model, test_loader, device=DEVICE) #<---------- Check this out
    # evaluate() runs the model on the test set and calculates metrics like accuracy, AUC, and loss.
    # It returns a dictionary of metrics that summarize the model's performance on the test set.
    print("\nTest-set metrics:")
    for k, v in metrics.items():
        print(f"  {k:>8s}: {v:.4f}") #prints the resulting metrics in a formatted way.

    # 6. Save final model (already saved best weights inside `train()`)
    with open("model/feature_names.pkl", "wb") as f:
    # Crucial when you deploy the model: saving `feature_names`
    # preserves the exact column order the network was trained on.
    # Because we one-hot encode categorical variables, the final
    # feature matrix is a wide array of 0/1 flags.  If you feed
    # the model the columns in a different order later, every
    # weight will attach to the wrong input and predictions will
    # be garbage.
    #
    # One-hot encoding = turning each category into its own binary
    # column (e.g. Sex_M, Sex_F).  This lets a neural net handle
    # categorical data just like numeric features.
        pickle.dump(feature_names, f)
    torch.save(model.state_dict(), "model/best_heartnet.pth")
    print("\nAll artefacts saved - ready for inference!")
    # Artefacts are all the files yu keep after training to reproduce the model later.
    # Inference means using a trained model to make predictions on new data.