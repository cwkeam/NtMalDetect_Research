import glob
import os

fromPath = './allMAL/'
endPath = './sysMAL/'


for filename in glob.glob(os.path.join(fromPath, '*.txt')):
    f = open(filename, 'r')
    newF = open(endPath + filename[9:], 'w')
    print(filename)
    for line in f:
        if line[:2] == 'Nt':
            newL = line.split('(')[0]
            newF.write(newL+'\n')
    newF.close()
    f.close()

