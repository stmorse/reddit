"""
Purpose of this is to recreate key functionality of the sentence-transformers package
By using the HF models directly

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1
"""

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# mean Pooling - take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # first element of model_output contains all token embeddings
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).\
        expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentenceTransformerHF:
    def __init__(self, model='all-MiniLM-L6-v1', device='cuda'):
        # load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v1')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v1')
        
        self.device = device
        self.model.to(self.device)

    def encode(self, sentences):
        # Tokenize sentences and put on GPU
        encoded_input = self.tokenizer(sentences, 
                                       padding=True, 
                                       truncation=True, 
                                       return_tensors='pt').to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

if __name__=="__main__":
    # test sentences
    sentences = ['This is an example sentence', 'I love examples']

    print('Loading model...')
    model = SentenceTransformerHF()

    print('Sentences: ', sentences)
    print('Encoding sentences... ')
    embeddings = model.encode(sentences)

    print('Embedding shape: ', embeddings.shape)
    print('Similarity: ', F.cosine_similarity(embeddings[0], embeddings[1], dim=0))
    