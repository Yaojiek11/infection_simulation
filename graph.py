import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("外出自粛なし")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 40, 41)
S = [ 990, 979, 960, 946, 920, 900, 861, 815, 757, 707, 651,
           604, 563, 527, 510, 502, 496, 495, 494, 494, 494,
           494, 494, 494, 494, 494, 494, 494, 494, 494, 494,
           494, 494, 494, 494, 494, 494, 494, 494, 494, 494]

E = [   0,  11,  30,  39,  59,  69,  95, 127, 158, 181, 203,
           215, 198, 185, 138, 102,  70,  38,  29,  17,   7,  
             4,   1,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

I = [   10,  10,  10,  13,  19,  24,  36,  44,  65,  79,  98,
           112, 141, 174, 200, 212, 192, 180, 143, 106,  77,
            51,  40,  25,  15,   8,   4,   2,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

R = [   0,   0,   0,   2,   2,   6,   7,  13,  19,  29,  42,
            58,  86, 101, 134, 163, 215, 254, 299, 344, 381,
           408, 422, 438, 446, 453, 455, 457, 458, 458, 458,
           458, 458, 458, 458, 458, 458, 458, 458, 458, 458]

D = [   0,   0,   0,   0,   0,   1,   1,   1,   1,   4,   6,
            11,  12,  13,  18,  21,  27,  33,  35,  39,  41,
            43,  43,  43,  45,  45,  47,  47,  48,  48,  48,
            48,  48,  48,  48,  48,  48,  48,  48,  48,  48]

plt.plot(x, S, color="blue", label="健康")
plt.plot(x, E, color="gold", label="潜伏")
plt.plot(x, I, color="red", label="発症")
plt.plot(x, R, color="green", label="回復")
plt.plot(x, D, color="black", label="死亡")

plt.legend()

plt.show()
fig.savefig("test1.png")
