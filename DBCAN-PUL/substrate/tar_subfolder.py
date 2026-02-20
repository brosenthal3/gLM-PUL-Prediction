import sys

import os

## for each line, there is a file

count = 1
for line in open(sys.argv[1]):
    lines = line.strip("\n").split("*")
    substrate = lines[-1]
    allfiles = " ".join(lines[0:len(lines)-1])
    print (allfiles,substrate)
    count += 1
    os.system('tar zcvf "' + substrate +  '".tar.gz ' + allfiles)
