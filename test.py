import tltk
dot = tltk.nlp.th2read('กลม')[1]
def testSplit(str):
    str = tltk.nlp.th2read(str)
    s = str.split('-')
    for i in range(len(s)):
        fd = s[i].find(dot)
        #print(fd)
        if fd != -1:
            if s[i][fd+1] in 'รลว' and s[i][fd-1] != 'ห':
                return -1
        s[i]=s[i].replace(dot,'',999)
    s.pop()
    return s.copy()
    
a = testSplit('สวัสดีวันจันทร์วันนี้วันสุขสบาย')
print(a)
a = testSplit('กราบ')
print(a)
a = testSplit('หวาย')
print(a)

if tltk.nlp.th2read('ก่ั่ว') == '':
    print('ff')
# print(tltk.nlp.th2read('ผมอยู่คุณ'))
# for i in tltk.nlp.th2read('อยู่'):
#     print(i)
    