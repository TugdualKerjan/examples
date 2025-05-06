from convnet import VGG11
from dataset import get_dataloaders
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", True)


BATCH_SIZE = 2
LEARNING_RATE = 0.01
WEIGHT_DECAY = 4e-4
EPOCHS = 100

key1 = jax.random.PRNGKey(0)
model = VGG11(key = key1)

train_dataloader, val_dataloader = get_dataloaders(BATCH_SIZE)

optimizer = optax.adam(LEARNING_RATE)

opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def loss(model, x, y):
    res = jax.vmap(model)(x)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(res, jax.nn.one_hot(y, 1000)))

def step(model, opt_state, optimizer: optax.GradientTransformation, x, y):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

for epoch in range(0, EPOCHS):
    
    for x, y in tqdm(train_dataloader):
        x, y = x.numpy(), y.numpy()
        model, opt_state, loss_value = step(model, opt_state, optimizer, x, y)
        print(f"Train Ministep: Loss: {loss_value}")

    for x, y in val_dataloader:
        x, y = x.numpy(), y.numpy()
        loss_value = loss(model, x, y)
        print(f"Eval Ministep: Loss: {loss_value}")

    print("-"*30)
    print(f"End of epoch {epoch}")