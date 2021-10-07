f = open('log.txt', 'r')
lines = f.readlines()
lines = lines[1::2]

fw = open('log_filted.txt','w+')
    
for line in lines:
    print(line, end='')
    fw.write(line)

fw.close()
