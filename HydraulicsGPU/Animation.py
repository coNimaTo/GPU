import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_frame(path):
    with open(path, 'rb') as f:
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        return np.frombuffer(f.read(), dtype=np.float32).reshape(N, N)

step_size = 50
total_steps = 10000
terrain_frame = load_frame(f'frames/terrain_0000.bin')
frames = [load_frame(f'frames/water_{i:04d}.bin') for i in range(0, 10000, step_size)]

fig, ax = plt.subplots(1,2, figsize = (12,4))
imt = ax[0].imshow(terrain_frame, cmap='terrain')
im  = ax[1].imshow(frames[0], cmap='Blues', vmin=0, vmax=1)
plt.colorbar(imt, ax=ax[0])
plt.colorbar(im, ax=ax[1], )

def update(i):
    im.set_data(frames[i])
    ax[1].set_title(f'Step {i*step_size}')
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=300/len(frames))
plt.show()