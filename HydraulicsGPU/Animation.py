import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_frame(path):
    with open(path, 'rb') as f:
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        return np.frombuffer(f.read(), dtype=np.float32).reshape(N, N)

frames = [load_frame(f'frames/water_{i:04d}.bin') for i in range(0, 1000, 50)]

fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='Blues')
plt.colorbar(im)

def update(i):
    im.set_data(frames[i])
    ax.set_title(f'Step {i*50}')
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
plt.show()