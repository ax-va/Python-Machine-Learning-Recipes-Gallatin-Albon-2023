"""
Automatically select the best hyperparameters for a neural network.
->
Use the Ray tuning library with PyTorch.

The Ray tuning library provides a sophisticated API
to schedule experiments on both CPUs and GPUs.

Install:
$ pip install "ray[tune]"
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create data with 10 features and 1000 observations
features, target = make_classification(
    n_classes=2,
    n_features=10,
    n_samples=1000,
    random_state=1,
)
# Because we are using simulated data using Scikit-Learn make_classification,
# we don't have to standardize the features.
# But in for real data, we must do standardization.

# Split training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target,
    test_size=0.1,
    random_state=1,
)
features_train.shape
# (900, 10)
features_test.shape
# (100, 10)

# Set random seed for PyTorch
torch.manual_seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)


# Define a neural network using "Sequential"
class SequentialNN(nn.Module):
    """
    Feedforward two-layer neural network for binary classification using nn.Sequential.
    Each layer is "dense" (also called "fully connected")
    = All the units in the previous layer and in the next layer are connected.
    """
    def __init__(self, layer_size_1=16, layer_size_2=16):
        """ Initiates a network architecture. """
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, layer_size_1),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size_1, layer_size_2),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size_2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sequential(x)
        return x


config = {
    "layer_size_1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "layer_size_2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
}

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=1000,
    grace_period=1,
    reduction_factor=2
)

cli_reporter = CLIReporter(
    parameter_columns=["layer_size_1", "layer_size_2", "lr"],
    metric_columns=["loss"]
)


# Train neural network
def train_model(config, num_epochs=3):
    network = SequentialNN(config["layer_size_1"], config["layer_size_2"])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(network.parameters(), lr=config["lr"], momentum=0.9)
    # Wrap data in TensorDataset
    train_data = TensorDataset(x_train, y_train)
    # Create data loader
    train_loader = DataLoader(
        train_data,
        # batch_size = number of observations to propagate
        # through the network before updating the parameters
        batch_size=100,
        shuffle=True,  # schlurfen, mischen
    )

    # Compile the model using torch 2.0's optimizer
    network = torch.compile(network)

    for epoch_idx in range(1, num_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html
            train.report({"loss": loss.item()})


result = tune.run(
    train_model,
    resources_per_trial={"cpu": 2},
    config=config,
    num_samples=1,
    scheduler=scheduler,
    progress_reporter=cli_reporter,
)

best_trial = result.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
best_trained_model = SequentialNN(
    best_trial.config["layer_size_1"],
    best_trial.config["layer_size_2"],
)
"""
2024-08-19 16:18:57,682	INFO worker.py:1781 -- Started a local Ray instance.
2024-08-19 16:18:58,263	INFO tune.py:253 -- Initializing Ray automatically. For cluster usage or custom Ray initialization, call `ray.init(...)` before `tune.run(...)`.
2024-08-19 16:18:58,282	WARNING tune.py:902 -- AIR_VERBOSITY is set, ignoring passed-in ProgressReporter for now.
╭────────────────────────────────────────────────────────────────────╮
│ Configuration for experiment     train_model_2024-08-19_16-18-58   │
├────────────────────────────────────────────────────────────────────┤
│ Search algorithm                 BasicVariantGenerator             │
│ Scheduler                        AsyncHyperBandScheduler           │
│ Number of trials                 1                                 │
╰────────────────────────────────────────────────────────────────────╯

View detailed results here: ...
To visualize your results with TensorBoard, run: `tensorboard --logdir /tmp/ray/...`

Trial status: 1 PENDING
Current time: 2024-08-19 16:18:58. Total running time: 0s
Logical resource usage: 2.0/8 CPUs, 0/0 GPUs
╭──────────────────────────────────────────────────╮
│ Trial name                status              lr │
├──────────────────────────────────────────────────┤
│ train_model_f96d1_00000   PENDING    0.000169444 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 started with configuration:
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 config             │
├──────────────────────────────────────────────────┤
│ layer_size_1                                 256 │
│ layer_size_2                                 128 │
│ lr                                       0.00017 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 1 at 2024-08-19 16:19:06. Total running time: 7s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         5.84466 │
│ time_total_s                             5.84466 │
│ training_iteration                             1 │
│ loss                                     0.68601 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 2 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00353 │
│ time_total_s                             5.84819 │
│ training_iteration                             2 │
│ loss                                      0.6875 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 3 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00226 │
│ time_total_s                             5.85045 │
│ training_iteration                             3 │
│ loss                                     0.69092 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 4 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00304 │
│ time_total_s                             5.85349 │
│ training_iteration                             4 │
│ loss                                     0.69165 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 5 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00233 │
│ time_total_s                             5.85583 │
│ training_iteration                             5 │
│ loss                                     0.68094 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 6 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                          0.0021 │
│ time_total_s                             5.85793 │
│ training_iteration                             6 │
│ loss                                     0.68648 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 7 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00207 │
│ time_total_s                                5.86 │
│ training_iteration                             7 │
│ loss                                     0.68855 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 8 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00384 │
│ time_total_s                             5.86384 │
│ training_iteration                             8 │
│ loss                                     0.68946 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 9 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00349 │
│ time_total_s                             5.86733 │
│ training_iteration                             9 │
│ loss                                       0.689 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 10 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00408 │
│ time_total_s                             5.87141 │
│ training_iteration                            10 │
│ loss                                     0.68551 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 11 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00341 │
│ time_total_s                             5.87482 │
│ training_iteration                            11 │
│ loss                                     0.68493 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 12 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00352 │
│ time_total_s                             5.87834 │
│ training_iteration                            12 │
│ loss                                     0.69308 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 13 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00383 │
│ time_total_s                             5.88217 │
│ training_iteration                            13 │
│ loss                                     0.68312 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 14 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                          0.0035 │
│ time_total_s                             5.88567 │
│ training_iteration                            14 │
│ loss                                     0.68709 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 15 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00358 │
│ time_total_s                             5.88925 │
│ training_iteration                            15 │
│ loss                                     0.68605 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 16 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00328 │
│ time_total_s                             5.89254 │
│ training_iteration                            16 │
│ loss                                     0.68148 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 17 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00399 │
│ time_total_s                             5.89653 │
│ training_iteration                            17 │
│ loss                                     0.69313 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 18 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00357 │
│ time_total_s                              5.9001 │
│ training_iteration                            18 │
│ loss                                     0.68455 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 19 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00382 │
│ time_total_s                             5.90391 │
│ training_iteration                            19 │
│ loss                                     0.68909 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 20 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00338 │
│ time_total_s                             5.90729 │
│ training_iteration                            20 │
│ loss                                     0.68703 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 21 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00341 │
│ time_total_s                              5.9107 │
│ training_iteration                            21 │
│ loss                                     0.68276 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 22 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00342 │
│ time_total_s                             5.91412 │
│ training_iteration                            22 │
│ loss                                      0.6828 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 23 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00329 │
│ time_total_s                             5.91741 │
│ training_iteration                            23 │
│ loss                                     0.68723 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 24 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00353 │
│ time_total_s                             5.92094 │
│ training_iteration                            24 │
│ loss                                      0.6819 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 25 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00198 │
│ time_total_s                             5.92292 │
│ training_iteration                            25 │
│ loss                                      0.6827 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 26 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00195 │
│ time_total_s                             5.92488 │
│ training_iteration                            26 │
│ loss                                     0.68765 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 finished iteration 27 at 2024-08-19 16:19:06. Total running time: 8s
╭──────────────────────────────────────────────────╮
│ Trial train_model_f96d1_00000 result             │
├──────────────────────────────────────────────────┤
│ checkpoint_dir_name                              │
│ time_this_iter_s                         0.00195 │
│ time_total_s                             5.92683 │
│ training_iteration                            27 │
│ loss                                     0.68043 │
╰──────────────────────────────────────────────────╯

Trial train_model_f96d1_00000 completed after 27 iterations at 2024-08-19 16:19:06. Total running time: 8s

Trial status: 1 TERMINATED
Current time: 2024-08-19 16:19:06. Total running time: 8s
Logical resource usage: 2.0/8 CPUs, 0/0 GPUs
╭───────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                status                lr     iter     total time (s)       loss │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ train_model_f96d1_00000   TERMINATED   0.000169444       27            5.92683   0.680426 │
╰───────────────────────────────────────────────────────────────────────────────────────────╯

2024-08-19 16:19:06,458	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/home/...' in 0.0030s.
"""
# Best trial config: {'layer_size_1': 256, 'layer_size_2': 128, 'lr': 0.0001694444389910084}
# Best trial final validation loss: 0.6804264187812805
