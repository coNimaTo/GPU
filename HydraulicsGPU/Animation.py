import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_frame(path):
    with open(path, 'rb') as f:
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        return np.frombuffer(f.read(), dtype=np.float32).reshape(N, N)

step_size = 10
total_steps = 5000
t_frames = [load_frame(f'frames/terrain_{i:05d}.bin') for i in range(0, total_steps, step_size)]
w_frames = [load_frame(f'frames/water_{i:05d}.bin') for i in range(0, total_steps, step_size)]
s_frames = [load_frame(f'frames/sediment_{i:05d}.bin') for i in range(0, total_steps, step_size)]

fig, ax = plt.subplots(1,3, figsize = (12,4))
t_im = ax[0].imshow(t_frames[0], cmap='terrain')
w_im  = ax[1].imshow(w_frames[0], cmap='Blues', vmin=0, vmax=.03)
s_im  = ax[2].imshow(t_frames[0]-t_frames[0], vmin=-0.5, vmax=.5)
plt.colorbar(t_im, ax=ax[0])
plt.colorbar(w_im, ax=ax[1])
plt.colorbar(s_im, ax=ax[2])

print(t_frames[0].max(),t_frames[0].min())

def update(i):
    t_im.set_data(t_frames[i])
    w_im.set_data(w_frames[i])
    s_im.set_data(t_frames[i]-t_frames[0])
    # s_im.set_data(s_frames[i])
    ax[1].set_title(f'Step {i*step_size}')
    return [t_im, w_im, s_im]

ani = animation.FuncAnimation(fig, update, frames=len(t_frames), interval=300/len(t_frames))
plt.show()