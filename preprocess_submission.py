import numpy as np
import data_handling_and_preparation as dhap


filename = "/home/tadek/Coding/Kaggle/SeaLionPopulation/sub.csv"

lines = dhap.read_csv(filename)

sol = np.array(lines).astype(np.float32)
sol = sol.astype(np.int32)

sol = sol[sol[:, 0].argsort()]

n, m = sol.shape

f = open("preprocessed_sub.csv", "w")
f.write("test_id,adult_males,subadult_males,adult_females,juveniles,pups\n")

count = 0
for i in range(n):
    #if (sol[i,0] != count):
    #    print("Breaking at: " + str(i))
    #    break
    #count = count + 1

    a = str(sol[i,0])
    b = str(sol[i,1])
    c = str(sol[i,2])
    d = str(sol[i,3])
    e = str(sol[i,4])
    g = str(sol[i,5])


    f.write(a + "," + b + "," + c + "," + d + "," + e + "," + g + "\n")

f.close()
