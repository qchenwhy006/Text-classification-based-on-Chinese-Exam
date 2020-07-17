import pandas as pd
import pathlib
import os
import re
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import jieba
from gensim.models.word2vec import Word2Vec,LineSentence


root = pathlib.Path(os.path.abspath(__file__)).parent.parent

def build_data(root):
    grade_labels = ['高中']
    subject_labels = ['地理', '历史', '生物', '政治']
    subdivision_labels = {'地理': ['地球与地图', '区域可持续发展', '人口与城市', '生产活动与地域联系', '宇宙中的地球'],
                          '历史': ['古代史', '近代史', '现代史'],
                          '生物': ['分子与细胞', '生物技术实践', '生物科学与社会', '稳态与环境', '现代生物技术专题', '遗传与进化'],
                          '政治': ['公民道德与伦理常识', '经济学常识', '科学社会主义常识', '科学思维常识', '生活中的法律常识', '时事政治']}
    df_target=pd.DataFrame(columns=['labels','content'])

    for grade in grade_labels:
        for subject in subject_labels:
            for subdivision in subdivision_labels[subject]:

                path=os.path.join(root,'data','百度题库','百度题库','高中_'+subject,'origin',subdivision+'.csv')
                raw_data=pd.read_csv(path)

                df_label=pd.DataFrame(columns=['labels','content'])
                df_label['labels']=raw_data['item'].apply(lambda x:[grade,subject,subdivision]+get_knowledge_point(x))
                df_label['content']=raw_data['item']
                df_target=df_target.append(df_label)

    min_samples=300
    labels=[]
    for item in df_target['labels']:
        labels.extend(item)
    labels=dict(pd.Series(labels).value_counts())
    selected_labels=[i[0] for i in list(labels.items())if i[1]>min_samples]
    print('selected labels:',len(selected_labels))

    df_target['labels']=df_target['labels'].apply(lambda x:x[:3]+list((set(x)-set(x[:3]))&set(selected_labels)))
    df_target['labels']=df_target['labels'].apply(lambda x:' '.join(x))
    path=os.path.join(root,'data','Y.csv')
    df_target['labels'].to_csv(path,index=None,header=False,encoding='utf-8')


    df_target['labels'] = df_target['labels'].apply(lambda x: x.split())
    mlb=MultiLabelBinarizer()
    Y=mlb.fit_transform(df_target['labels'])
    path=os.path.join(root,'data','Y')
    np.save(path,Y)

    # Build up embedding_matirx
    df_target['content']=df_target['content'].apply(lambda x:x.replace('[题目]\n',''))
    df_target['content']=df_target['content'].apply(sentence_processing)
    path=os.path.join(root,'data','merged_data.csv')
    df_target['content'].to_csv(path,index=None,header=False,encoding='utf-8')

    print('start build w2v model')
    wv_model=Word2Vec(LineSentence(path),size=300,negative=5,workers=8,iter=10,window=3,min_count=5)
    vocab = wv_model.wv.vocab


    content_max_len=get_max_len(df_target['content'])
    df_target['content']=df_target['content'].apply(lambda x:pad_proc(x,content_max_len,vocab))
    path=os.path.join(root,'data','X.csv')
    df_target['content'].to_csv(path,index=None,header=False,encoding='utf-8')

    print('start retrain w2v model')
    wv_model.build_vocab(LineSentence(path),update=True)
    wv_model.train(LineSentence(path),epochs=10,total_examples=wv_model.corpus_count)
    path=os.path.join(root,'data','word2vec.model')
    wv_model.save(path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))

    vocab={word: index for index,word in enumerate(wv_model.wv.index2word)}
    reverse_vocab={index:word for index,word in enumerate(wv_model.wv.index2word)}
    word_to_vectors_path = os.path.join(root, 'data','word_to_vectors.txt')
    vectors_to_word_path = os.path.join(root, 'data', 'vectors_to_word.txt')
    with open(word_to_vectors_path,'w+',encoding='utf-8') as f:
        for k,v in vocab.items():
            f.write('{}\t{}\n'.format(k,v))
    with open(vectors_to_word_path,'w+',encoding='utf-8') as f:
        for k,v in reverse_vocab.items():
            f.write('{}\t{}\n'.format(k,v))


    embedding_matrix=wv_model.wv.vectors
    embedding_matrix_path = os.path.join(root, 'data','embedding_matrix')
    np.save(embedding_matrix_path, embedding_matrix)

    idx=df_target['content'].apply(lambda x:transform_data(x, vocab))
    X=np.array(idx.tolist())
    X_path = os.path.join(root,'data','X')
    np.save(X_path, X)


    return X,Y


def get_knowledge_point(sentence):
    index=sentence.find('[知识点：]')
    return re.split('、|,|，',sentence[index+7:])


def translate_to_labels(root,y_pred):
    path=os.path.join(root,'data','Y.csv')
    df=pd.read_csv(path,header=None,names=['labels'])
    df['labels']=df['labels'].apply(lambda x:x.split())
    mlb=MultiLabelBinarizer()
    mlb.fit_transform(df['labels'])

    return mlb.inverse_transform(y_pred)


def load_stop_words(root):
    path=os.path.join(root,'data','stopwords.txt')
    file=open(path,'r',encoding='utf-8')
    lines=file.readlines()
    stop_words=[line.strip() for line in lines]
    return stop_words


def sentence_processing(sentence):
    sentence=clean_sentence(sentence)
    words=[word for word in sentence if word not in stop_words]
    return ' '.join(words)


def clean_sentence(sentence):
    sentence=re.sub('[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）º℃°；>．]+|题目|①|②|③|④','',sentence)
    sentence=jieba.cut(sentence, cut_all=False)
    return sentence


def load_stop_words(root):
    path=os.path.join(root,'data','stopwords.txt')
    file=open(path,'r',encoding='utf-8')
    lines=file.readlines()
    stop_words=[line.strip() for line in lines]
    return stop_words


def get_max_len(dataframe):
    num=dataframe.apply(lambda x:x.count(' ')+1)
    return int(np.mean(num)+2*np.std(num))


def pad_proc(sentence,max_len,vocab):
    words=sentence.strip().split(' ')
    words=words[:max_len]
    sentence=[word if word in vocab else '<UNK>' for word in words]
    sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)


def transform_data(sentence,vocab):
    words=sentence.strip().split(' ')
    idx=[vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
    return idx


def load_data(root):
    path_x=os.path.join(root,'data','X')
    x=np.load(path_x + '.npy')
    path_y = os.path.join(root, 'data', 'Y')
    y=np.load(path_y + '.npy')
    return x,y


def load_embedding_matrix(root):
    path=os.path.join(root,'data','embedding_matrix')
    return np.load(path+'.npy')


def load_vocab(root):
    path = os.path.join(root, 'data', 'word_to_vectors.txt')
    file=open(path,'r',encoding='utf-8')
    lines=file.readlines()
    vocab={words.strip().split('\t')[0]:index for index,words in enumerate(lines)}
    return vocab


def load_and_process_data(data_dir, local_save=True, sample_min_num=300, clean=True, use_jieba=True):
    """

    :param data_dir: Local path which contains raw data
    :param local_save: Whether to save processed data in local storage
    :return: A DataFrame contains processed data
    """
    subjects = ['地理', '历史', '生物', '政治']
    # subjects = ['历史']

    processed_data = None
    for subject in subjects:
        subject_dir = os.path.join(data_dir, '高中_{}'.format(subject))
        file_names = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f))]
        for file_name in file_names:
            cur_processed_data = pd.DataFrame(columns=['id', 'item', 'subject', 'category', 'knowledge'])
            print('Start process {}'.format(os.path.join(subject_dir, file_name)))
            cur_raw_data = pd.read_csv(os.path.join(subject_dir, file_name), encoding='utf-8')
            cur_processed_data = cur_raw_data.apply(lambda x: data_loader_helper(x, clean=clean, use_jieba=use_jieba),
                                                    axis=1)
            cur_processed_data['id'] = cur_raw_data['web-scraper-order']
            cur_processed_data['subject'] = subject
            cur_processed_data['category'] = file_name.split('.')[0]

            if processed_data is None:
                processed_data = cur_processed_data
            else:
                processed_data = pd.concat([processed_data, cur_processed_data])

    appear_dict = dict()
    for knowledges in processed_data['knowledge']:
        for knowledge in knowledges:
            if knowledge in appear_dict.keys():
                appear_dict[knowledge] = appear_dict[knowledge] + 1
            else:
                appear_dict[knowledge] = 1

    processed_data['knowledge'] = processed_data['knowledge'].apply(
        lambda x: [k for k in x if appear_dict[k] >= sample_min_num])
    processed_data['knowledge'] = processed_data['knowledge'].apply(lambda x: ' '.join(x) if len(x) > 0 else np.nan)
    processed_data['labels'] = processed_data.apply(lambda x:
                                                    ' '.join(
                                                        [x['subject'], x['category'], x['knowledge']]) if not pd.isnull(
                                                        x['knowledge'])
                                                    else ' '.join([x['subject'], x['category']]), axis=1)

    if local_save:
        saved_path = 'data/total_processed_data.csv'
        processed_data.to_csv(saved_path, index=False, encoding='utf-8')
    return processed_data

def data_loader_helper(xdat, clean=True, use_jieba=True):
    knowledge, item = process_text(xdat['item'], clean=clean, use_jieba=use_jieba)
    xdat['knowledge'] = knowledge
    xdat['processed_item'] = item
    return xdat[['processed_item', 'knowledge']]



if __name__ == '__main__':
    root = pathlib.Path(os.path.abspath(__file__)).parent
    stop_words = load_stop_words(root)
    X,Y=build_data(root)
    embedding_matrix=load_embedding_matrix(root)
    vocab=load_vocab(root)