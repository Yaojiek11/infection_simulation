import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("健康")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 30, 31)
S = [ 990, 948, 731, 461, 358, 331, 327, 327, 327, 327, 327,
           327, 327, 327, 327, 327, 327, 327, 327, 327, 327,
           327, 327, 327, 327, 327, 327, 327, 327, 327, 327]

S_control = [ 990, 944, 733, 721, 706, 693, 685, 670, 661, 652, 643,
           639, 635, 633, 632, 632, 630, 627, 626, 625, 624,
           624, 624, 624, 624, 624, 624, 624, 624, 624, 624]

plt.plot(x, S, color="blue", label="外出自粛なし")
plt.plot(x, S_control, color="blue", label="外出自粛あり", linestyle="dashdot")

plt.legend()

plt.show()
fig.savefig("S.png")