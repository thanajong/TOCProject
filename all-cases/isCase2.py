# เคส สระเสียงสั้น
# 15/10/23 แก้ใส่สระเสียงยาวแล้วไม่ออกลูป,แก้สระเ-า เพิ่ม *สระ-ุ* *สระเ-ียะ* *สระเ-ือะ* *สระ-ัวะ* *สระ-ำ*
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
            elif(a[i] in 'ุ'):
                qf = True
                i+=1
            elif(a[i] in 'ำ'):
                qf=True
                i+=1
            else: i+=1
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
                elif(a[i] in 'ว' and i==len(a)-1):
                    i+=1
                elif(a[i] in 'ว'):
                    i+=1
                    if(a[i] in 'ะ'):
                        qf=True 
                        i+=1
            elif(a[i] in 'กดบนงมย'):
                qf = True
                i+=1
            elif(a[i] in 'ว' and i==len(a)-1):
                i+=1
            elif(a[i] in 'ว'):
                i+=1
                if(a[i] in 'ะ'):
                    qf=True 
                    i+=1
        elif(a[i] in '-ิ-ึ'):
            qf = True
            i+=1
        elif(a[i] in 'ุ'):
            qf = True
            i+=1
        elif(a[i] in 'ำ'):
            qf=True
            i+=1
        else: i+=1
    elif(a[i] in 'โแ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ' and i == len(a)-1):
            i+=1
        elif(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            i+=1
            if(a[i] in '-่-้-๊-๋' and i == len(a)-1):
                i+=1
            elif(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'ะ'):
                    qf = True
                    i+=1
            elif(a[i] in 'ะ'):
                qf = True
                i+=1
    elif(a[i] in 'เ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ' and i == len(a)-1):
            i+=1
        elif(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            i+=1
            if(a[i] in '-่-้-๊-๋' and i == len(a)-1):
                i+=1
            elif(a[i] in '-่-้-๊-๋'):
                i+=1
                if(a[i] in 'ะ'):
                    qf = True
                    i+=1
                elif(a[i] in '็'):
                    i+=1
                    if(a[i] in 'กดบนงม'):
                        qf = True
                        i+=1
                elif(a[i] in 'า' and (i == len(a)-1)):
                    qf = True
                    i+=1
                elif(a[i] in 'า'):
                    i+=1
                    if(a[i] in 'ะ'):
                        qf = True
                        i+=1
                elif(a[i] in 'อ' and i==len(a)-1):
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
            elif(a[i] in 'า' and (i == len(a)-1)):
                qf = True
                i+=1
            elif(a[i] in 'า'):
                i+=1
                if(a[i] in 'ะ'):
                    qf = True
                    i+=1
            elif(a[i] in 'อ' and i==len(a)-1):
                i+=1
            elif(a[i] in 'อ'):
                i+=1
                if(a[i] in 'ะ'):
                    qf = True
                    i+=1
            elif(a[i] in 'ี'):
                i+=1
                if(a[i] in '่้๊๋'):
                    i+=1
                    if(a[i] in 'ย' and i==len(a)-1):
                        i+=1
                    elif(a[i] in 'ย'):
                        i+=1
                        if(a[i] in 'ะ'):
                            i+=1
                            qf = True
                elif(a[i] in 'ย' and i==len(a)-1):
                    i+=1
                elif(a[i] in 'ย'):
                    i+=1
                    if(a[i] in 'ะ'):
                        i+=1
                        qf = True
            elif(a[i] in 'ื'):
                i+=1
                if(a[i] in '่้๊๋'):
                    i+=1
                    if(a[i] in 'อ' and i==len(a)-1):
                        i+=1
                    elif(a[i] in 'อ'):
                        i+=1
                        if(a[i] in 'ะ'):
                            i+=1
                            qf = True   
                elif(a[i] in 'อ' and i==len(a)-1):
                    i+=1
                elif(a[i] in 'อ'):
                    i+=1
                    if(a[i] in 'ะ'):
                        i+=1
                        qf = True      
    elif(a[i] in 'ไใ'):
        i+=1
        if(a[i] in 'กขคงจฉชซดตถทนบปผฝพฟมยรลวสหอฮ'):
            qf = True
            i+=1
    while(i<=(len(a)-1)):
        if( (qf==True) and (a[i] in '-่-้-๊-๋กดบนงมยว') ):
            i+=1
        else: i+=1
    return qf
             

word = 'กิ้'
print(isCase2(word))