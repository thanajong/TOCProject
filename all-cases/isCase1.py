# เคส สระเสียงยาว
# 14/10/23 ดักเคสสระเสียงสั้นไม่ออกลูป   *เพิ่มสระ-ู*   *แโ ไม่มีตัวสะกด*

def isCase1(a):
    i=0
    qf = False
    if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
        i+=1
        if(a[i] in '-่-้-๊-๋'):
            i+=1
            if(a[i] in 'ู'):
                qf=True
                i+=1
            elif(a[i] in 'ั'):
                i+=1
                if(a[i] in '-่-้-๊-๋'):
                    i+=1
                    if(a[i] in 'ว'):
                        qf = True
                        i+=1
                        if((i<=len(a)-1) and a[i] in 'ะ'):
                            qf = False
                            i+=1
                elif(a[i] in 'ว'):
                    qf = True
                    i+=1
                    if((i<=len(a)-1) and a[i] in 'ะ'):
                            qf = False
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
            else: i+=1
        elif(a[i] in 'ู'):
                qf=True
                i+=1
        elif(a[i] in 'ั'):
            i+=1
            if(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'ว'):
                    qf = True
                    i+=1
                    if((i<=len(a)-1) and a[i] in 'ะ'):
                        qf = False
                        i+=1
                else: i+=1
            elif(a[i] in 'ว'):
                qf = True
                i+=1
                if((i<=len(a)-1) and a[i] in 'ะ'):
                        qf = False
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
        else: 
            for i in range(len(a)):
                i+=1
    elif(a[i] in 'เ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            qf = True
            i+=1
            if((i<=len(a)-1) and a[i] in '่้๊๋'):
                i+=1
                if((i<=len(a)-1) and a[i] in 'ะ'):
                    qf = False
                    i+=1
                elif((i<=len(a)-1) and a[i] in 'อ'):
                    i+=1
                    if((i<=len(a)-1) and a[i] in 'ะ'):
                        qf = False
                        i+=1
                elif((i<=len(a)-1) and a[i] in 'า'):
                    qf = False
                    i+=1
                    if((i<=len(a)-1) and a[i] in 'ะ'):
                        qf = False
                        i+=1
            if((i<=len(a)-1) and a[i] in 'ะ'):
                qf = False
                i+=1
            elif((i<=len(a)-1) and a[i] in 'อ'):
                i+=1
                if((i<=len(a)-1) and a[i] in 'ะ'):
                    qf = False
                    i+=1
            elif((i<=len(a)-1) and a[i] in 'า'):
                qf = False
                i+=1
                if((i<=len(a)-1) and a[i] in 'ะ'):
                    qf = False
                    i+=1
    elif(a[i] in 'แโ'):
        i+=1
        # if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ' and (i!=len(a)-1)):
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):         
            qf=True
            i+=1
            if((i<=len(a)-1) and a[i] in '่้๊๋'):
                i+=1
                if((i<=len(a)-1) and a[i] in 'กดบนงมยว'):
                    qf = True
                    i+=1 
                elif((i<=len(a)-1) and a[i] in 'ะ'):
                    qf = False
                    i+=1
            elif((i<=len(a)-1) and a[i] in 'กดบนงมยว'):
                qf = True
                i+=1
            elif((i<=len(a)-1) and a[i] in 'ะ'):
                qf = False
                i+=1
            else: i+=1
        elif( (a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ') and (i==len(a)-1) ):
            qf=True
            i+=1 
    else:
        for i in range(len(a)):
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
                        if((i<=len(a)-1) and a[i] in 'ะ'):
                            qf = False
                            i+=1
                elif(a[i] in 'ย'):
                    i+=1
                    if((i<=len(a)-1) and a[i] in 'ะ'):
                        qf = False
                        i+=1
            elif(a[i] in '-ื'):
                i+=1
                if(a[i] in '-่-้-๊-๋'):
                    i+=1
                    if(a[i] in 'อ'):
                        i+=1
                        if((i<=len(a)-1) and a[i] in 'ะ'):
                            qf = False
                            i+=1
                elif(a[i] in 'อ'):
                        i+=1
                        if((i<=len(a)-1) and a[i] in 'ะ'):
                            qf = False
                            i+=1
    return qf   

word = 'โก'
print(isCase1(word))