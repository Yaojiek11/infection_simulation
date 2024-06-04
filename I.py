import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("発症")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 30, 31)
I = [  10,  10,  10,  20,  98, 212, 283, 346, 405, 400, 334,
           240, 155,  92,  45,  12,   2,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

I_control = [  10,  10,  10,  20,  94, 105, 155, 173, 193, 181, 116,
           104,  59,  54,  41,  34,  31,  27,  17,  13,   8,
             7,   7,   5,   4,   3,   1,   0,   0,   0,   0]

plt.plot(x, I, color="red", label="外出自粛なし")
plt.plot(x, I_control, color="red", label="外出自粛あり", linestyle="dashdot")

plt.legend()

plt.show()
fig.savefig("I.png")