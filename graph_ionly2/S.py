import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()

plt.title("健康状態：S")
plt.grid()
plt.xlabel("経過日数[日]")
plt.ylabel("人数[人]")

x = np.linspace(0, 60, 61)
S1 = [ 990, 974, 966, 944, 929, 900, 846, 784, 705, 641, 579,
           535, 518, 512, 504, 503, 501, 500, 500, 500, 500,
           500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
           500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
           500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
           500, 500, 500, 500, 500, 500, 500, 500, 500, 500]

S2 = [ 990, 974, 966, 948, 933, 912, 861, 819, 758, 694, 629,
           626, 619, 618, 615, 607, 598, 595, 590, 587, 585,
           582, 578, 575, 573, 572, 568, 568, 568, 567, 567,
           566, 566, 565, 565, 565, 564, 564, 564, 564, 564,
           564, 564, 564, 564, 564, 564, 564, 564, 564, 564,
           564, 564, 564, 564, 564, 564, 564, 564, 564, 564]

S3 = [ 990, 973, 965, 945, 929, 888, 844, 792, 788, 784, 782,
           778, 770, 761, 752, 744, 735, 728, 724, 720, 716,
           711, 709, 703, 700, 695, 695, 695, 692, 691, 690,
           690, 689, 689, 686, 683, 681, 680, 677, 675, 675,
           675, 674, 674, 672, 671, 671, 671, 671, 671, 671,
           670, 670, 670, 670, 669, 669, 669, 669, 669, 669]

S4 = [ 990, 974, 966, 944, 921, 874, 871, 869, 863, 862, 858,
           854, 848, 842, 839, 833, 832, 824, 818, 812, 809,
           806, 805, 802, 800, 798, 798, 796, 796, 794, 792,
           790, 787, 786, 785, 784, 784, 784, 782, 782, 782,
           782, 782, 779, 778, 778, 778, 778, 775, 774, 774,
           773, 772, 772, 772, 771, 771, 770, 770, 768, 768]

plt.plot(x, S1, color="blue", label="外出自粛なし")
plt.plot(x, S2, color="blue", label="外出自粛あり(10日目以降)", linestyle="dashed")
plt.plot(x, S3, color="blue", label="外出自粛あり(7日目以降)", linestyle="dashdot")
plt.plot(x, S4, color="blue", label="外出自粛あり(5日目以降)", linestyle="dotted")

plt.legend()

plt.show()
fig.savefig("S.png")