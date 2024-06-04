import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("死亡")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 30, 31)
D = [   0,   0,   0,   0,   0,   1,   6,  14,  20,  27,  36,
            44,  55,  65,  70,  73,  74,  74,  74,  74,  74,
            74,  74,  74,  74,  74,  74,  74,  74,  74,  74]

D_control = [   0,   0,   0,   0,   0,   1,   5,   8,  12,  13,  20,
            25,  35,  37,  40,  40,  40,  41,  42,  42,  43,
            43,  43,  43,  43,  43,  43,  44,  44,  44,  44]

plt.plot(x, D, color="black", label="外出自粛なし")
plt.plot(x, D_control, color="black", label="外出自粛あり", linestyle="dashdot")

plt.legend()

plt.show()
fig.savefig("D.png")