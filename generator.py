import matplotlib.pyplot as plt
from models import generator, seed

predictions = generator(seed, training=False)
plt.figure(num=1, figsize=(4, 4))

for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
    plt.axis("off")

plt.show()
