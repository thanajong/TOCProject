# เคส สระเสียงยาว
def isCase1(a):
    i=0
    qf = False
    if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
        i+=1
        if(a[i] in '-่-้-๊-๋'):
            i+=1
            if(a[i] in 'ั'):
                i+=1
                if(a[i] in '-่-้-๊-๋'):
                    i+=1
                    if(a[i] in 'ว'):
                        qf = True
                        i+=1
                elif(a[i] in 'ว'):
                    qf = True
                    i+=1
            elif(a[i] in 'วอา-ี'):
                qf = True
                i+=1
            elif(a[i] in 'ว'):
                i+=1
                if(a[i] in 'กดบนงมย'):
                    qf = True
                    i+=1
            elif(a[i] in '-ื'):
                i+=1
                if(a[i] in '-่-้-๊-๋'):
                    i+=1
                    if(a[i] in 'กดบนงมอ'):
                        qf=True
                        i+=1
                elif(a[i] in 'กดบนงมอ'):
                    qf=True
                    i+=1
        elif(a[i] in 'ั'):
            i+=1
            if(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'ว'):
                    qf = True
                    i+=1
            elif(a[i] in 'ว'):
                qf = True
                i+=1
        elif(a[i] in 'วอา-ี'):
            qf = True
            i+=1
        elif(a[i] in 'ว'):
            i+=1
            if(a[i] in 'กดบนงมย'):
                qf = True
                i+=1
        elif(a[i] in '-ื'):
            i+=1
            if(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'กดบนงมอ'):
                    qf=True
                    i+=1
            elif(a[i] in 'กดบนงมอ'):
                qf=True
                i+=1
    elif(a[i] in 'เ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            qf = True
            i+=1
    elif(a[i] in 'แโ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            i+=1
            if(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'กดบนงมยว'):
                    qf = True
                    i+=1 
            elif(a[i] in 'กดบนงมยว'):
                qf = True
                i+=1 
    while(i<=(len(a)-1)):
        if(qf==True):
            if(a[i] in '-่-้-๊-๋กดบนงมยว'):
                i+=1
            elif(a[i] in 'ิ'):
                i+=1
                if(a[i] in '-่-้-๊-๋'):
                    i+=1
                    if(a[i] in 'กดบนงม'):
                        i+=1
                elif(a[i] in 'กดบนงม'):
                    i+=1
            elif(a[i] in 'ี'):
                i+=1
                if(a[i] in '-่-้-๊-๋'):
                    i+=1
                    if(a[i] in 'ย'):
                        i+=1
                elif(a[i] in 'ย'):
                    i+=1
            elif(a[i] in '-ื'):
                i+=1
                if(a[i] in '-่-้-๊-๋'):
                    i+=1
                    if(a[i] in 'อ'):
                        i+=1
                elif(a[i] in 'อ'):
                        i+=1
    return qf   

word = 'เมื่อ'
print(isCase1(word))