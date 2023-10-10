# เคส สระเสียงสั้น
def isCase2(a):
    i=0
    qf = False
    if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
        i+=1
        if(a[i] in '-่-้-๊-๋'):
            i+=1
            if(a[i] in 'กดบนงม'):
                qf = True
                i+=1
            elif(a[i] in 'ะ'):
                qf = True
                i+=1
            elif(a[i] in 'ั'):
                i+=1
                if(a[i] in '-่-้-๊-๋'):
                    i+=1
                    if(a[i] in 'กดบนงมย'):
                        qf = True
                        i+=1
                elif(a[i] in 'กดบนงมย'):
                    qf = True
                    i+=1
            elif(a[i] in '-ิ-ึ'):
                qf = True
                i+=1
        elif(a[i] in 'กดบนงม'):
            qf = True
            i+=1
        elif(a[i] in 'ะ'):
            qf = True
            i+=1
        elif(a[i] in 'ั'):
            i+=1
            if(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'กดบนงมย'):
                    qf = True
                    i+=1
            elif(a[i] in 'กดบนงมย'):
                qf = True
                i+=1   
        elif(a[i] in '-ิ-ึ'):
            qf = True
            i+=1
    elif(a[i] in 'โแ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            i+=1
            if(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'ะ'):
                    qf = True
                    i+=1
            elif(a[i] in 'ะ'):
                qf = True
                i+=1
    elif(a[i] in 'เ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            i+=1
            if(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'ะ'):
                    qf = True
                    i+=1
                elif(a[i] in '็'):
                    i+=1
                    if(a[i] in 'กดบนงม'):
                        qf = True
                        i+=1
                elif(a[i] in 'า' and not(a[i+1] in 'ะ')):
                    qf = True
                    i+=1
                elif(a[i] in 'า'):
                    i+=1
                    if(a[i] in 'ะ'):
                        qf = True
                        i+=1
                elif(a[i] in 'อ'):
                    i+=1
                    if(a[i] in 'ะ'):
                        qf = True
                        i+=1
            elif(a[i] in 'ะ'):
                qf = True
                i+=1
            elif(a[i] in '็'):
                i+=1
                if(a[i] in 'กดบนงม'):
                    qf = True
                    i+=1
            elif(a[i] in 'า' and not(a[i+1] in 'ะ')):
                qf = True
                i+=1
            elif(a[i] in 'า'):
                i+=1
                if(a[i] in 'ะ'):
                    qf = True
                    i+=1
            elif(a[i] in 'อ'):
                i+=1
                if(a[i] in 'ะ'):
                    qf = True
                    i+=1
    elif(a[i] in 'ไใ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            qf = True
            i+=1
    while(i<=(len(a)-1)):
        if( (qf==True) and (a[i] in '-่-้-๊-๋กดบนงมยว') ):
            i+=1
    return qf
             

word = 'ไร'
print(isCase2(word))