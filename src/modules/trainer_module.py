import typing as tp

# JAX/Flax/Optax
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints

# Loggers
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger


class TrainState(train_state):
    '''
    Any functionality to keep track of
    '''
    batch_stats: tp.Any = None
    rng: tp.Any = None
    
    
class TrainerModule:
    def __init__(self,
                 model_class: nn.Module,
                 model_hparams: tp.Dict[str, tp.Any],
                 optimizer_hparams: tp.Dict[str, tp.Any],
                 dummy_input: tp.Any,
                 seed: int = 42,
                 logger_params: tp.Dict[str, tp.Any] = None,
                 progress_bar: bool = True,
                 debug: bool = False,
                 check_val_every_n_epoch: int = 1,
                 **kwargs):
        '''
        Training Module in the same fashion as in Pytorch Lightning Trainer class
        '''
        
        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.progress_bar = progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.dummy_input = dummy_input
        
        self.config = {
            'model_class': model_class.__name__,
            'model_hparams': model_hparams,
            'optimizer_hparams': optimizer_hparams,
            'logger_params': logger_params,
            'progress_bar': progress_bar,
            'debug': self.debug,
            'check_val_every_n_epoch': check_val_every_n_epoch,
            'seed': self.seed
        }
        
        self.config.update(kwargs)
        self.model = self.model_class(**self.model_hparams)
        self.print_model(dummy_input)
        
    def print_model(self, dummy_input: tp.Any):
        print(self.model.tabulate(jax.random.PRNGKey(0), *dummy_input, train=True))
        



