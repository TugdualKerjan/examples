from convnet import VGG11
from dataset import get_dataloaders
import jax

import equinox as eqx
import optax


import logging
import logging.config

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("universalLogger")



BATCH_SIZE = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 4e-4
EPOCHS = 100

key1 = jax.random.PRNGKey(0)
model = VGG11(key = key1)

train_dataloader, val_dataloader = get_dataloaders(BATCH_SIZE)

optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(model)

@eqx.filter_jit
def loss(model, x, y):
    res = jax.vmap(model)(x)
    return optax.sigmoid_binary_cross_entropy(res, y)

def step(model, opt_state, optimizer: optax.GradientTransformation, x, y):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss_value

for epoch in range(0, EPOCHS):
    for i, x, y, _ in enumerate(train_dataloader):
        x, y = x.numpy(), y.numpy()
        model, opt_state, loss_value = step(model, opt_state, optimizer, x, y)
        logger.info(f"Train Ministep: {i}, Loss: {loss_value}")

    for x, y, _ in val_dataloader:
        x, y = x.numpy(), y.numpy()
        loss_value = loss(model, x, y)
        logger.info(f"Eval Ministep: {i}, Loss: {loss_value}")

    logger.info("-"*30)
    logger.info(f"End of epoch {epoch}")