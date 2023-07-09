import re
import time

import requests
import json
'''
weiBoDate=[
    "2019_12","2019_12","2019_12","2019_12","2019_12",
    "2020_3","2020_3","2020_3","2020_3","2020_3",
    "2020_6","2020_6","2020_6","2020_6","2020_6",
    "2020_9","2020_9","2020_9","2020_9","2020_9",
    "2020_12","2020_12","2020_12","2020_12","2020_12",
    "2021_3","2021_3","2021_3","2021_3","2021_3",
    "2021_6","2021_6","2021_6","2021_6","2021_6",
    "2021_9","2021_9","2021_9","2021_9","2021_9",
    "2021_12","2021_12","2021_12","2021_12","2021_12",
    "2022_3","2022_3","2022_3","2022_3","2022_3",
    "2022_6", "2022_6", "2022_6", "2022_6", "2022_6",
    "2022_9", "2022_9", "2022_9", "2022_9", "2022_9",
    "2022_12","2022_12","2022_12","2022_12","2022_12"
           ]
weiBoID=[
    "4466905887135419","4466896252155552","4466520220377390","4466407645080319","4465982250774920",
    "4485927374647820","4485924672705820","4487929747460574","4487023974423820","4486998163704132",
    "4515703761965042","4515995006758283","4520698360080273","4516353869890606","4516107929773947",
    "4548956195916441","4546423830086033","4547150544115042","4546066001170321","4544611421258417",
    "4587412414529803","4586004730355062","4584841033290641","4584244918359898","4583754884973585",
    "4618910961764937","4616135775097371","4616119711958931","4614697427142134","4616384959218198",
    "4646607858961328","4644638809392827","4643258804207693","4648247773103944","4646514985276482",
    "4681249602406982","4681097663481625","4683833578884196","4683068470199193","4682680950587632",
    "4715858348474737","4715439550892697","4710431224498989","4713535613698427","4711902645846199",
    "4750091020143369","4748657209116547","4747302981600244","4745480598194240","4745393701392531",
    "4785609292650637","4779932356707499","4779574720202298","4779208847660393","4778852952052619",
    "4809540044328883","4809177065066350","4809901953515905","4810988786026831","4817874373577527",
    "4847640930161141","4846118250745808","4849480845231548","4849018699516448","4848293975163864"
]'''

weiBoDate=[
    "2020_9", "2020_9", "2020_9", "2020_9", "2020_9","2020_9", "2020_9",
    "2022_6", "2022_6", "2022_6", "2022_6", "2022_6","2022_6", "2022_6",
    "2022_9", "2022_9", "2022_9", "2022_9", "2022_9","2022_9", "2022_9"
           ]
weiBoID=[
    "4539762772027580","4539687454377449","4538461425506238","4538099159794699","4537737984347062",
    "4537370042959222","4537094414542522",
    "4771610122782807","4771235017524290","4769692298250076","4768367699297145","4766528232492583",
    "4762772803683780","4762410516217952",
    "4811374733824447","4811554300369925","4800076448531947","4779932356707499","4779574720202298",
    "4779208847660393","4778847964237237"
]


weiBoUid="1784473157"
#weiBoDate=["test"]
#weiBoID=[4920152015249098]
#weiBoUid=[2656274875]
def DataClean(string):
    re_tag = re.compile(r'(\s{2,})|(@)|((http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?)|(<[a-zA-Z]+.*?>)|(</[a-zA-Z]+.*?>)|(#.*?#)|(回复.*?:)')
    return re_tag.sub('', string)

def getPackage(weiboID,uid,max_id,is_mix="0",fetch_level="0"):
    url = "https://weibo.com/ajax/statuses/buildComments"
    headers = {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "cookie":"UOR=www.baidu.com,weibo.com,www.baidu.com; SINAGLOBAL=6280917346990.507.1687004294789; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFv-JrIKvFEKHncP6TgPy_i5JpX5KMhUgL.Foq0ShBN1KBfe0q2dJLoIEXLxKMLB.2LBKzLxK.LB.eLB.zLxK-LB-BL1K5LxK-LBo5L12qLxKqLBoeLBKzt; ariaDefaultTheme=default; ariaFixed=true; ariaReadtype=1; ariaMouseten=null; ariaStatus=false; XSRF-TOKEN=yR3fmpzI2Lc07ng6MRvd366I; ALF=1691463537; SSOLoginState=1688871539; SCF=AuzoF6-gxgeHZOcuHDrZoWEi9Hg0z8mAQ57D0cjvborxaC3YBMFNlcrFsQQdkC374epnebbIXusDjyzmuBNN920.; SUB=_2A25JrlIjDeRhGeBN71YW-SrJyDqIHXVq2sTrrDV8PUNbmtANLXL8kW9NRGam754Fd6-Qb1UxeYCxpKSMWTAGfvXB; _s_tentry=weibo.com; Apache=4972053713055.797.1688871545737; ULV=1688871545766:6:5:1:4972053713055.797.1688871545737:1688699977864; WBPSESS=E0nptfb99a_mEoN719yyXoIZgDd4g2NxLEi_2mhnJ58Lug48L_8V3QbUlxc9OYz-vx2mxFRymHLHDhkiql2IGEEipL9piNJuNWUaEZWb8ZufH9kDWYKIGRDcdn9_CvfPMlJFqmP75MxdFFJBPxPqHg=="
    }
    params = {
        "flow":"1",
        "is_reload": "1",
        "id": weiboID,
        "is_show_bulletin": "2",
        "is_mix": is_mix,
        "fetch_level": fetch_level,
        "count": "10",
        "uid": uid

    }
    if max_id!=None:
        params['max_id'] = max_id
        params['count'] = "20"
    retryTimes = 0
    while True:
        r = requests.get(url, headers=headers, params=params,timeout=10)
        if r.status_code==200: #如果获取失败则重新尝试
            break
        retryTimes=retryTimes+1
        print("retry "+str(retryTimes)+"....")
        time.sleep(0.01)
    try:
        obj=r.json()
    except:
        print("==============Error===============")
        print(r.status_code)
        print(obj)
    r.close()
    return obj

def WriteTextToFile(path,text):
    f = open(path, mode='a', encoding='utf-8')
    f.write(text)
    f.close()

def WriteReplay(weiBoDate,weiBoId,weiBoUid):
    max_id=0
    count = 0
    batchString = ""
    NullPackageNumber=0
    while True:
        time.sleep(0.01)
        packageSub = getPackage(weiBoId,weiBoUid,max_id,is_mix="1",fetch_level="1")
        data = packageSub['data']
        max_id = str(packageSub['max_id'])
        print("get a package...\tmaxid:"+str(max_id)+"\tdata length:" + str(len(data)))
        if len(data)==0:
            NullPackageNumber=NullPackageNumber+1
            if NullPackageNumber > 20 or max_id == "0":
                break
            continue
        for item in data:
            #if userlist.__contains__(str(item['user']['id'])) == True:#使用布隆过滤器防止重复评论
                #continue
            #userlist[str(item['user']['id'])] = 1

            createInfo = item['created_at'].split(' ')
            if int(createInfo[-1]) > int(weiBoDate.split('_')[0])+1: #筛选创建微博一年内的评论
                continue

            dataString = item['text']
            dataString = dataString.strip()
            dataString = DataClean(dataString)
            if JudgeChineseLen(dataString, 3) == False: #筛选大于三个字的评论
                continue
            dataString = str(item['user']['id']) + "\t" + dataString + "\n"
            batchString = batchString + dataString
            count = count + 1
            if count % 100 == 0:
                print("\nfinished a batch....!\n")
                WriteTextToFile('WeiBoComment_' + weiBoDate + '.txt',batchString)
                count = 0
                batchString = ""
        if max_id == "0":
            break
    WriteTextToFile('WeiBoComment_' + weiBoDate + '.txt', batchString)
    print("====================== a replay finished !!!!=========================")
def JudgeChineseLen(text,wordNumber):
    count=0
    for s in text:
        if s.isalpha():
           count=count+1
        if count>=wordNumber:
            return True
    if count <wordNumber:
        return False

def ProjectRun():
    for index in range(len(weiBoID)):
        max_id = None
        count = 0
        batchString = ""
        nullPackageTimes=0
        while True:
            time.sleep(0.01)
            package = getPackage(weiBoID[index], weiBoUid, max_id)
            data = package["data"]
            max_id = str(package["max_id"])
            print("get a package...\tmaxid:" + str(max_id) + "\tdata length:" + str(len(data)))
            #if len(data) == 0 and max_id == "0":
                #break
            if len(data) == 0 and max_id != "0":
                nullPackageTimes=nullPackageTimes+1
                if nullPackageTimes > 50:
                    break
                continue
            for item in data:
                dataString  =   item['text']
                dataString = dataString.strip()
                dataString = DataClean(dataString)
                if JudgeChineseLen(dataString,3)==False:
                    continue
                dataString = str(item['user']['id']) + "\t" + dataString + "\n"
                batchString = batchString + dataString
                if int(item['total_number']) > 0:
                    WriteReplay(weiBoDate[index],item['id'], weiBoUid)
                count = count+1
                if count % 100 == 0:
                    print("finished a batch....!")
                    count = 0
                    WriteTextToFile('WeiBoComment_' + weiBoDate[index] + '.txt', batchString)
                    batchString = ''
            if max_id == "0":
                break
        WriteTextToFile('WeiBoComment_' + weiBoDate[index] + '.txt', batchString)
        print("finished file:"+str(index+1))

ProjectRun()