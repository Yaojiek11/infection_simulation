import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("死亡状態：D")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 60, 61)
D1 = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             1,   1,   1,   1,   2,   2,   3,   4,   9,  10,
            13,  17,  21,  25,  34,  35,  42,  43,  45,  45,
            46,  47,  47,  47,  47,  47,  47,  47,  47,  47,
            47,  47,  47,  47,  47,  47,  47,  47,  47,  47,
            47,  47,  47,  47,  47,  47,  47,  47,  47,  47]

D2 = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             1,   1,   2,   2,   2,   2,   5,   7,   9,  10,
            12,  16,  19,  22,  26,  31,  32,  36,  37,  38,
            38,  38,  38,  39,  39,  39,  39,  39,  39,  40,
            40,  42,  42,  42,  42,  42,  42,  42,  42,  42,
            42,  42,  42,  42,  42,  42,  42,  42,  42,  42]

D3 = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             1,   1,   2,   2,   3,   4,   5,   7,   9,  10,
            10,  14,  17,  17,  21,  22,  23,  24,  24,  24,
            27,  27,  27,  28,  28,  28,  28,  31,  31,  31,
            31,  32,  32,  32,  33,  33,  33,  34,  34,  35,
            35,  36,  36,  37,  37,  37,  37,  37,  37,  37]

D4 = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             1,   1,   1,   1,   3,   3,   4,   4,   7,   7,
             7,   8,   9,   9,  10,  10,  10,  11,  11,  12,
            12,  12,  12,  12,  13,  13,  13,  13,  14,  14,
            14,  14,  14,  14,  14,  14,  15,  15,  15,  15,
            15,  15,  16,  16,  16,  16,  16,  16,  16,  16]

plt.plot(x, D1, color="black", label="外出自粛なし")
plt.plot(x, D2, color="black", label="外出自粛あり(10日目以降)", linestyle="dashed")
plt.plot(x, D3, color="black", label="外出自粛あり(7日目以降)", linestyle="dashdot")
plt.plot(x, D4, color="black", label="外出自粛あり(5日目以降)", linestyle="dotted")

plt.legend()

plt.show()
fig.savefig("D.png")