import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def load_frame(path):
    with open(path, 'rb') as f:
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        return np.frombuffer(f.read(), dtype=np.float32).reshape(N, N)

step_size = 10
total_steps = 5000

t1 = time.time()
t_frames = [load_frame(f'frames/terrain_{i:05d}.bin') for i in range(0, total_steps, step_size)]
t2 = time.time()
print(f"Terrain loaded in {(t2-t1):.3f}s")

w_frames = [load_frame(f'frames/water_{i:05d}.bin') for i in range(0, total_steps, step_size)]
t1 = time.time()
print(f"Water loaded in {(t1-t2):.3f}s")

s_frames = [load_frame(f'frames/sediment_{i:05d}.bin') for i in range(0, total_steps, step_size)]

fig, ax = plt.subplots(1,3, figsize = (12,4))
w_im  = ax[1].imshow(w_frames[0], cmap='Blues', vmin=0, vmax=.03)

# t_im = ax[0].imshow(t_frames[0], cmap='terrain')
diff_im  = ax[0].imshow(t_frames[0]-t_frames[0], cmap='terrain', vmin=-0.1, vmax=.1)
s_im     = ax[2].imshow(t_frames[0]-t_frames[0], cmap='terrain', vmin=-.05, vmax=.05)

for axx in ax:
    axx.set(xticks= [], yticks=[])

plt.colorbar(w_im, ax=ax[1])

# plt.colorbar(t_im, ax=ax[0])
plt.colorbar(diff_im, ax=ax[0])
plt.colorbar(s_im, ax=ax[2])

# print(t_frames[0].max(),t_frames[0].min())
t2 = time.time()
print(f"Initial plot crated in {(t2-t1):.3f}s")

def update(i):
    # t_im.set_data(t_frames[i])
    diff_im.set_data(t_frames[i]-t_frames[0])
    w_im.set_data(w_frames[i])
    s_im.set_data(s_frames[i])
    # s_im.set_data(s_frames[i])
    ax[1].set_title(f'Step {i*step_size}')
    return [diff_im, w_im, s_im]

ani = animation.FuncAnimation(fig, update, frames=len(t_frames), interval=300/len(t_frames))
ani.save('animation.mp4', writer='ffmpeg', fps=30)
plt.show()
t1 = time.time()
print(f"Animation made in {(t1-t2):.3f}s")