
def transform(num):
    res = []

    while num!=0:
        i = num%2
        num = num//2
        res.append(i)
    print("原数据：",res)
    i = 0
    if len(res)%2==0:
        leng = len(res)
    else:
        leng = len(res) - 1
    while i<leng:
        res[i],res[i+1] = res[i+1],res[i]
        i += 2
    print("新数据：",res)
    new_num = 0
    for i in range(len(res)-1,-1,-1):
        new_num += res[i]*2**(len(res)-1-i)
    print("输出：",new_num)

transform(585)




