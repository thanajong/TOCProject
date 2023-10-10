#case สระเสียง อู
def isCase9(a):
    i=0
    qf=False
    if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
        i+=1
        if(a[i]=='ู'):
            qf=True
            i+=1
    while(i<=(len(a)-1)):
        if( (qf==True) and (a[i] in '-่-้-๊-๋กดบนงมยว') ):
            i+=1
        else: 
            qf = False
            i+=1
    return qf

word = 'มู่น'
print(isCase9(word))