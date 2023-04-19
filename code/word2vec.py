import gensim
import codecs


#파일을 열어 각 줄을 토큰화한 리스트를 반환
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()

#word2vec 모델은 gensim의 Word2Vec을 사용. 
def main(domain):
    source = '../preprocessed_data/%s/train.txt' % (domain)
    model_file = '../preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4)
    model.save(model_file)


print()
main('restaurant')
main('beer')



