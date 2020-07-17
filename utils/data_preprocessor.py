import pandas as pd
import os
import pathlib
import re
from collections import Counter
import numpy as np


def data2file(file_dir,type='csv'):
    grades=['高中']
    subjects=['地理', '历史', '生物', '政治']
    categories={'地理':['地球与地图','宇宙中的地球','生产活动与地域联系','人口与城市','区域可持续发展'],
                '历史':['古代史', '近代史', '现代史'],
                '生物':['现代生物技术专题','生物科学与社会','生物技术实践','稳态与环境','遗传与进化','分子与细胞'],
                '政治':['经济学常识', '科学思维常识', '生活中的法律常识','科学社会主义常识','公民道德与伦理常识','时事政治']}

    df_target=pd.DataFrame(columns=['label','item'])
    for grade in grades:
        for subject in subjects:
            for category in categories[subject]:
                file=os.path.join(file_dir,grade + '_' + subject,'origin',category+'.csv')
                df=pd.read_csv(open(file,encoding='utf-8'))
                print('{} {} {} size:{}'.format(grade,subject,category,len(df)))

                df['item']=df['item'].apply(lambda x:''.join(x.split()))
                df['label']=df['item'].apply(lambda x:[grade,subject,category]+re.split('、|,|，',x[x.find('[知识点：]')+7:]))
                df['item']=df['item'].apply(lambda x: x.replace('[题目]',''))
                df['item']=df['item'].apply(lambda x: x[:x.index('题型')] if x.index('题型') else x)


                df=df[['label','item']]
                df_target=df_target.append(df)
    print('origin data size:',len(df_target))

    min_samples=300
    df=df_target.copy()
    labels=[]
    for i in df['label']:
        labels.extend(i)

    result=dict(sorted(dict(Counter(labels)).items(),key=lambda x:x[1],reverse=True))
    lens=np.array(list(result.values()))
    Label_num=len(lens[lens>min_samples])

    label_target=set([k for k,v in result.items() if v>min_samples])

    df['label']=df['label'].apply(lambda x: x[:3]+list(set(x)-set(x[:3]) & label_target))
    df['label']=df['label'].apply(lambda x: None if len(x)<4 else x) # Remove the row without knowledge point
    df=df[df['label'].notna()]

    # Confirm the final label number
    labels=[]
    [labels.extend(i) for i in df['label']]
    Label_num=len(set(labels))

    print('datasize:{}, multi_class:{}'.format(len(df),Label_num))
    df['label']=df['label'].apply(lambda x:' '.join(x))



    file=os.path.join(file_dir,f'baidu_{Label_num}.csv')
    df.to_csv(file, index=False, sep=',', header=False, encoding='UTF8')











if __name__ == '__main__':
    root=pathlib.Path(os.path.abspath(__file__)).parent
    file_dir=os.path.join(root,'百度题库','百度题库')
    data2file(file_dir)