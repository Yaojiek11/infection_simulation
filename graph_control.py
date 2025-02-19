import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("外出自粛あり")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 40, 41)

S = [ 990, 977, 959, 945, 921, 901, 859, 796, 741, 678, 675,
           674, 668, 664, 662, 658, 653, 652, 652, 652, 651,
           650, 650, 650, 650, 650, 650, 650, 650, 650, 650,
           650, 650, 650, 650, 650, 650, 650, 650, 650, 650]

E = [    0, 13,  31,  42,  57,  69,  95, 140, 174, 206, 167,
           120,  97,  53,  30,  15,  17,  13,   8,   7,   5,  
             4,   2,   1,   1,   1,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

I = [  10,  10,  10,  11,  20,  24,  37,  49,  67,  87, 106,
           129, 137, 157, 146, 124, 102,  70,  57,  30,  17,
            11,  10,   8,   2,   2,   2,   2,   2,   1,   1,
             1,   0,   0,   0,   0,   0,   0,   0,   0,   0]

R = [   0,   0,   0,   2,   2,   4,   7,  13,  16,  25,  46,
            68,  88, 114, 145, 183, 205, 239, 256, 283, 297,
           303, 306, 309, 315, 315, 315, 315, 315, 315, 315,
           315, 315, 315, 315, 315, 315, 315, 315, 315, 315]

D = [   0,   0,   0,   0,   0,   2,   2,   2,   2,   4,   6,
             9,  10,  12,  17,  20,  23,  26,  27,  28,  30,
            32,  32,  32,  32,  32,  33,  33,  33,  33,  33,
            33,  33,  33,  33,  33,  33,  33,  33,  33,  33]

plt.plot(x, S, color="blue", label="健康")
plt.plot(x, E, color="gold", label="潜伏")
plt.plot(x, I, color="red", label="発症")
plt.plot(x, R, color="green", label="回復")
plt.plot(x, D, color="black", label="死亡")

plt.legend()

plt.show()
fig.savefig("test2.png")
