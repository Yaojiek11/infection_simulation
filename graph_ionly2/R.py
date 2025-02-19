import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("回復状態：R")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 60, 61)
R1 = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             2,   2,   7,   8,  22,  33,  51,  64,  87, 118,
           150, 195, 232, 275, 321, 361, 385, 417, 430, 439,
           445, 449, 450, 452, 452, 453, 453, 453, 453, 453,
           453, 453, 453, 453, 453, 453, 453, 453, 453, 453,
           453, 453, 453, 453, 453, 453, 453, 453, 453, 453]

R2 = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             2,   3,  10,  10,  20,  23,  34,  53,  69,  89,
           126, 169, 194, 229, 253, 289, 311, 332, 339, 359,
           365, 369, 373, 376, 379, 383, 384, 387, 389, 389,
           389, 389, 390, 390, 391, 391, 393, 393, 393, 393,
           393, 393, 393, 394, 394, 394, 394, 394, 394, 394]

R3 = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             2,   2,   8,  10,  23,  26,  43,  55,  83,  95,
           121, 136, 160, 173, 190, 201, 209, 216, 219, 225,
           234, 239, 243, 248, 252, 258, 262, 264, 269, 270,
           273, 273, 275, 276, 276, 276, 276, 276, 277, 278,
           280, 282, 284, 286, 287, 287, 287, 289, 290, 291]

R4 = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             2,   2,   8,  10,  23,  31,  56,  63,  79,  89,
           102, 110, 121, 126, 131, 132, 136, 140, 147, 151,
           157, 158, 164, 169, 176, 179, 181, 183, 184, 187,
           188, 189, 190, 191, 192, 193, 195, 196, 198, 200,
           200, 201, 201, 201, 201, 202, 205, 205, 205, 207]

plt.plot(x, R1, color="green", label="外出自粛なし")
plt.plot(x, R2, color="green", label="外出自粛あり(10日目以降)", linestyle="dashed")
plt.plot(x, R3, color="green", label="外出自粛あり(7日目以降)", linestyle="dashdot")
plt.plot(x, R4, color="green", label="外出自粛あり(5日目以降)", linestyle="dotted")

plt.legend()

plt.show()
fig.savefig("R.png")