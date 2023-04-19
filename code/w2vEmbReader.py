import codecs
import logging
import numpy as np
import gensim
from sklearn.cluster import KMeans
import pickle  


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class W2VEmbReader:
    #gensim 라이브러리를 사용해 사전 학습된 단어 임베딩을 로드, 
    #단어가 키이고 임베딩 벡터가 값인 사전에 저장. 
    def __init__(self, emb_path, emb_dim=None):

        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []
       
        model = gensim.models.Word2Vec.load(emb_path)
        self.emb_dim = emb_dim
        for word in model.wv.key_to_index:#vocab->wv.vocab, 또 오류 나서 model.wv.key_to_index으로 바꿈
            self.embeddings[word] = list(model.wv[word])
            emb_matrix.append(list(model.wv[word]))

        if emb_dim is not None:#!= 을 is not 으로 바꿈.
            assert self.emb_dim == len(self.embeddings['숙소'])#nice를 숙소로 바꿈
            
        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))

    #단어를 입력으로 받고 해당 단어의 임베딩 벡터를 반환, 
    #단어가 사전 학습된 임베딩에서 찾을 수 없는 경우 None을 반환
    def get_emb_given_word(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None
    
    #vocab 사전과 빈 numpy 배열을 입력으로 받아 numpy 배열을 vocab 사전에 있는 임베딩으로 채움
    #vocab 사전에서 단어가 사전학습된 임베딩에서 찾을 수 없는 경우 해당 단어를 건너뜀. 
    #이 메서드는 정규화된 임베딩 numpy 배열을 반환한다. 
    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():#iteritems()에서 items로 바꿈
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix
    
    #사전학습된 임베딩을 K-means 클러스터링해 유사한 단어를 그룹화하고 클러스터 중심으로 aspect matrix로 반환. 
    #반환하기 전 aspect matrix를 정규화. 
    def get_aspect_matrix(self, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_

        # L2 normalization
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        return norm_aspect_matrix.astype(np.float32)
    
    #사전학습 된 임베딩의 차원을 반환. 
    def get_emb_dim(self):
        return self.emb_dim
    
    
    
    
