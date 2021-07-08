from dataset import *

data = get_dataset()

savepic(data["QPSK", 18][0], 2, 128, "test")

PS1 = power_spectrum(data["QPSK", 2][0][0])
PS2 = power_spectrum(data["QPSK", 2][0][1])
PS = (PS1 + PS2) / 2
print(PS)

savepic(PS, 1, 128, "PS")