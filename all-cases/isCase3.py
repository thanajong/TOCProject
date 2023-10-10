#case ร,ล
def isCase3(a):
    i = 0
    qf = False
    if(a[i] in 'รล'): 
        qf = True
        i+=1
    elif(a[i] in 'เแโ'): 
        i+=1 
        if(a[i] in 'รล'):
            qf = True
            i+=1
    while(i<=(len(a)-1)):
        if( (qf==True) and (a[i] in '-่-้-๊-๋กดบนงมยวะ-ัอา-ี-ิ-ึ-ื-็') ):
            i+=1
        else: i+=1 # เช่น คำว่า ลูก
    return(qf)
            

word='ลูก'
print(isCase3(word))