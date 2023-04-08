import numpy as np
import sys
sys.path.append("/home/m_bobrin/PapersInJAX")

from torch.utils import data
from src.modules.data_module import create_data_loaders
from src.modules.trainer_module import TrainerModule
from typing import *
import flax.linen as nn
import jax

def target_function(x):
    return np.sin(x * 3.0)

class RegressionDataset(data.Dataset):

    def __init__(self, num_points, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(low=-2.0, high=2.0, size=num_points)
        self.y = target_function(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx:idx+1], self.y[idx:idx+1]

class MLPRegressor(nn.Module):
    hidden_dims : Sequence[int]
    output_dim : int

    @nn.compact
    def __call__(self, x, **kwargs):
        for dims in self.hidden_dims:
            x = nn.Dense(dims)(x)
            x = nn.silu(x)
        x = nn.Dense(self.output_dim)(x)
        return x
    
train_set = RegressionDataset(num_points=1000, seed=42)
val_set = RegressionDataset(num_points=200, seed=43)
test_set = RegressionDataset(num_points=500, seed=44)
train_loader, val_loader, test_loader = create_data_loaders(train_set, val_set, test_set,
                                                            train=[True, False, False],
                                                            batch_size=64)

class MLPRegressTrainer(TrainerModule):
    def __init__(self,
                 hidden_dims : Sequence[int],
                 output_dim : int,
                 **kwargs):
        super().__init__(model_class=MLPRegressor,
                         model_hparams={
                             'hidden_dims': hidden_dims,
                             'output_dim': output_dim
                         },
                         **kwargs)

    def create_functions(self):
        def mse_loss(params, batch):
            x, y = batch
            pred = self.model.apply({'params': params}, x)
            loss = ((pred - y) ** 2).mean()
            return loss

        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return state, metrics

        def eval_step(state, batch):
            loss = mse_loss(state.params, batch)
            return {'loss': loss}

        return train_step, eval_step
    
trainer = MLPRegressTrainer(hidden_dims=[128, 128],
                            output_dim=1,
                            optimizer_hparams={'lr': 4e-3},
                            logger_params={'base_log_dir': "/home/m_bobrin/PapersInJAX/checkpoints"},
                            dummy_input=next(iter(train_loader))[0:1],
                            check_val_every_n_epoch=5)

metrics = trainer.train_model(train_loader,
                              val_loader,
                              test_loader=test_loader,
                              num_epochs=50)

print(f'Validation accuracy: {metrics["val/acc"]:4.2%}')
print(f'Test accuracy: {metrics["test/acc"]:4.2%}')