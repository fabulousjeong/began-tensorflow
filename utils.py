import matplotlib.pyplot as plt
import numpy as np

# generate noise vector
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# generate UNIFORM [-1, 1] noise vector
def get_uniform_noise(batch_size, n_noise):
    return np.random.uniform(-1, 1, size=(batch_size, n_noise))

# visualize sample output
def save_samples(title, samples):
    samples = (samples+1.0)*0.5 # normalize to [0,1]
    n_grid = int(np.sqrt(samples.shape[0]))
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(2*n_grid, 2*n_grid))

    samples_grid = np.reshape(samples[:n_grid * n_grid],(n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))

    if samples.shape[3] != 3:
        samples_grid = np.squeeze(samples_grid, 4)

    for i in range(n_grid):
        for j in range(n_grid):
            axes[i][j].set_axis_off()
            axes[i][j].imshow(samples_grid[i][j])

    plt.savefig(title, bbox_inches='tight')
    print('saved %s.' % title)
    plt.close(fig)