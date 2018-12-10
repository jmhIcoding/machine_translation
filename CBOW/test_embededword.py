__author__ = 'jmh081701'
import  json
import numpy as np
npy_path = "data//input.en.txt.ebd.npy"
vab_path="data//input.en.txt.vab"
W=np.load(npy_path)
inverseV={}#map word to index
V={} #map index to word
with open(vab_path,encoding='utf8') as fp:
    V=json.load(fp)
    for each in V:
        inverseV.setdefault(V[each],int(each))
def find_min_word(word,k):
    vec = W[inverseV[word]]
    tmp=[]
    for i in range(len(V)):
        tmp.append(np.matmul(vec-W[i].T,(vec-W[i].T).T))
    s =[]
    rst=[]
    for i in range(len(tmp)):
        s.append([tmp[i],i])
    s.sort()
    print(s)
    for i in range(k):
        rst.append(V[str(s[i][1])])
    return rst
print(find_min_word('6',10))
