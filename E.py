import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("潜伏")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 30, 31)
E = [   0,  42, 259, 516, 541, 447, 353, 246, 134,  47,  12,
             1,   0,   0,   0,   0,   0,   0,   0,   0,   0,  
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

E_control = [   0,  46, 257, 256, 197, 191, 127, 113,  46,  40,  39,
            33,  29,  22,  11,   6,   4,   6,   6,   5,   3,  
             1,   0,   0,   0,   0,   0,   0,   0,   0,   0]

plt.plot(x, E, color="gold", label="外出自粛なし")
plt.plot(x, E_control, color="gold", label="外出自粛あり", linestyle="dashdot")

plt.legend()

plt.show()
fig.savefig("E.png")