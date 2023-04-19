from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs#문자열을 인코딩하고 디코딩하는데 사용되는 모듈을 불러옴. 


# 문장을 파싱하는 함수, 문장을 입력받아 불용어를 제거하고, 어간 추출, 그 결과를 반환
def parseSentence(line): 
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())#문장을 토큰화, 문자열을 소문자로 만듦
    text_rmstop = [i for i in text_token if i not in stop]#불용어 제거
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]#어간 추출
    return text_stem

#주어진 도메인의 훈련 데이터셋을 전처리하는 함수,
#  train.txt를 열어 파싱한 결과를 ../preprocessed_data/domain/train.txt 파일에 작성
def preprocess_train(domain):
    f = codecs.open('../datasets/'+domain+'/train.txt', 'r', 'utf-8')
    out = codecs.open('../preprocessed_data/'+domain+'/train.txt', 'w', 'utf-8')

    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n')

#주어진 도메인의 테스트 데이터셋을 전처리하는 함수
def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('../datasets/'+domain+'/test.txt', 'r', 'utf-8')
    f2 = codecs.open('../datasets/'+domain+'/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('../preprocessed_data/'+domain+'/test.txt', 'w', 'utf-8')
    out2 = codecs.open('../preprocessed_data/'+domain+'/test_label.txt', 'w', 'utf-8')
    
    #레스토랑 도메인인 경우 레이블이 food, staff, Ambience 중 하나인 것만 사용. +
    for text, label in zip(f1, f2):
        label = label.strip()
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        tokens = parseSentence(text)
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label+'\n')

def preprocess(domain):
    print('\t'+domain+' train set ...') 
    preprocess_train(domain)
    print('\t'+domain+' test set ...') 
    preprocess_test(domain)

print('Preprocessing raw review sentences ...') 
preprocess('restaurant')
preprocess('beer')


