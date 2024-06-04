import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("回復")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 30, 31)
R = [   0,   0,   0,   3,   3,   9,  31,  67, 114, 199, 291,
           388, 463, 516, 558, 588, 597, 599, 599, 599, 599,
           599, 599, 599, 599, 599, 599, 599, 599, 599, 599]

R_control = [   0,   0,   0,   3,   3,  10,  28,  36,  88, 114, 182,
           199, 242, 254, 276, 288, 295, 299, 309, 315, 322,
           325, 326, 328, 329, 330, 332, 332, 332, 332, 332]

plt.plot(x, R, color="green", label="外出自粛なし")
plt.plot(x, R_control, color="green", label="外出自粛あり", linestyle="dashdot")

plt.legend()

plt.show()
fig.savefig("R.png")