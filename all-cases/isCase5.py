#case !(ร or ล)
# 15/10/23 เพิ่มสระ ไใ-ุ-ู
def isCase5(a):
    i=0
    qf=False
    if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยวสหอฮ'):
        qf = True
        i+=1
    elif(a[i] in 'เแโไใ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยวสหอฮ'):
            qf = True
            i+=1
    while(i<=(len(a)-1)):
        if( (qf==True) and (a[i] in '-่-้-๊-๋กดบนงมยวะ-ัอา-ี-ิ-ึ-ื-็-ุ-ู') ):
            i+=1
        else: # จำเป็นต้องมี else นี้ไม่งั้นคำว่า เรือ จะไม่ออกจากลูป 
            qf = False
            i+=1
    return qf

word = 'เกือ'
print(isCase5(word))
