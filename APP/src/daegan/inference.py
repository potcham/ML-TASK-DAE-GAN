from src.daegan.miscc.config import cfg, cfg_from_file
from src.daegan.miscc.utils import weights_init
from src.daegan.model import G_NET
from src.daegan.model import RNN_ENCODER, CNN_ENCODER
from PIL import Image
# from torch.autograd import Variable
import numpy as np
import torch
import json
import os

from nltk.tokenize import RegexpTokenizer
import nltk

ROOT_STREAMLIT = 'src/daegan'
class TextEncoder:
    def __init__(self, train_word_to_idx):
        self.word_to_idx = train_word_to_idx
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        
    def preprocess_prompt(self, prompt):
        # text = 'the bird is dark grey brown with a thick curved bill and a flat shaped tail.'
        prompt = prompt.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(prompt.lower())

        sentence_tag = nltk.pos_tag(tokens)
        # CUB
        grammar = "NP: {<DT>*<JJ>*<CC|IN>*<JJ>+<NN|NNS>+|<DT>*<NN|NNS>+<VBZ>+<JJ>+<IN|CC>*<JJ>*}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(sentence_tag)

        attr_list = []
        for i in range(len(tree)):
            if type(tree[i]) == nltk.tree.Tree:
                attr = []
                for j in range(len(tree[i])):
                    attr.append(tree[i][j][0])
                attr_list.append(attr)

        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                tokens_new.append(t)
                
        captions = []
        for w in tokens_new:
            if w in self.word_to_idx:
                captions.append(self.word_to_idx[w])

        attrs = []
        for attr in attr_list:
            attr_new = []
            for w in attr:
                if w in self.word_to_idx:
                    attr_new.append(self.word_to_idx[w])
            attrs.append(attr_new)

        return prompt, captions, attrs # prompt, caption, attr
    
    def preprocess_caption(self, captions):
        sent_caption = np.asarray(captions).astype('int64')
        num_words = len(sent_caption)
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
            
        return x, x_len # caption with format, len(caption)
    
    def preprocess_attr(self, attrs):
        sen_attr = attrs # self.attrs[sent_ix]
        num_attrs = len(sen_attr)  # num of attr per sentence
        # cfg.MAX_ATTR_NUM, cfg.MAX_ATTR_LEN
        # pad with 0s (i.e., '<end>') ,  max_atr_len = 5
        sen_attr_new = []
        attr_cnt = 0
        # new
        sen_attr_new = np.zeros((cfg.MAX_ATTR_NUM, cfg.MAX_ATTR_LEN, 1), dtype='int64')

        for attr in sen_attr:
            attr = np.asarray(attr).astype('int64')
            # print(attr.shape, "====", attr)
            attr_cnt = attr_cnt + 1
            if attr_cnt > cfg.MAX_ATTR_NUM:
                break

            attr_len = len(attr)

            if attr_len <= cfg.MAX_ATTR_LEN:
                sen_attr_new[attr_cnt-1][:attr_len, 0] = attr
            else:
                ix = list(np.arange(attr_len))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:cfg.MAX_ATTR_LEN]
                ix = np.sort(ix)
                sen_attr_new[attr_cnt-1][:, 0] = attr[ix]

        return sen_attr_new
    
    def process_prompt(self, prompt):
        prompt, captions, attr_list = self.preprocess_prompt(prompt)
        captions, len_caption = self.preprocess_caption(captions)
        attributes = self.preprocess_attr(attr_list)
        
        return captions, len_caption, attributes
    

class DAE_GAN:
    def __init__(self, cfg_file, device='cpu'):
        self.cfg_file = cfg_file
        cfg_from_file(self.cfg_file )
        
        self.device = device
        self.N_WORDS = 5450
        self.BATCH_SIZE = 1

        self.wordx_to_idx = {}
        with open(os.path.join(ROOT_STREAMLIT,'word_to_ix.json')) as json_file:
            self.wordx_to_idx = json.load(json_file)
    
        self.netG = None
        self.text_encoder = None
        self.image_encoder = None
        self.noise = None

    def load_models(self):
        self.netG = G_NET()
        self.netG.apply(weights_init)
        cfg.TRAIN.NET_G = os.path.join(ROOT_STREAMLIT,cfg.TRAIN.NET_G)
        state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        print('Load G from: ', cfg.TRAIN.NET_G)
        self.netG.to(self.device)
        self.netG.eval()

        self.text_encoder = RNN_ENCODER(self.N_WORDS, nhidden=cfg.TEXT.EMBEDDING_DIM)
        cfg.TRAIN.NET_E = os.path.join(ROOT_STREAMLIT,cfg.TRAIN.NET_E)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        CNN_PRETRAIN = 'trained_models/inception_v3_google-1a9a5a14.pth'
        CNN_PRETRAIN = os.path.join(ROOT_STREAMLIT,CNN_PRETRAIN)
        self.image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM, pretrained_path = CNN_PRETRAIN)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        self.image_encoder.load_state_dict(state_dict)
        print('Load image encoder from:', img_encoder_path)
        self.image_encoder.to(self.device) 
        self.image_encoder.eval()

        nz = cfg.GAN.Z_DIM
        # self.noise = Variable(torch.FloatTensor(self.BATCH_SIZE, nz), volatile=True)
        self.noise = torch.FloatTensor(self.BATCH_SIZE, nz)
        self.noise.to(self.device, non_blocking=True)

        # return None
    
    def generate_img(self, captions, len_caption, attributes, save_dir = 'output/img_1.png'):
    
        captions = torch.from_numpy(captions).squeeze().unsqueeze(dim=0).to(self.device)
        cap_lens = torch.Tensor([len_caption]).type(torch.int64).to(self.device)
        attrs = torch.from_numpy(attributes).squeeze().unsqueeze(dim=0).to(self.device)
        
        hidden = self.text_encoder.init_hidden(self.BATCH_SIZE)
        words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        
        attr_len = torch.Tensor([cfg.MAX_ATTR_LEN] * cap_lens.size(0))
        _, attr_emb0 = self.text_encoder(attrs[:, 0:1, :].squeeze(dim=0), attr_len, hidden)
        _, attr_emb1 = self.text_encoder(attrs[:, 1:2, :].squeeze(dim=0), attr_len, hidden)
        _, attr_emb2 = self.text_encoder(attrs[:, 2:3, :].squeeze(dim=0), attr_len, hidden)
        attr_embs = torch.stack((attr_emb0, attr_emb1, attr_emb2), dim=2) 
        
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
            
        self.noise.data.normal_(0, 1)
        # print(self.noise.dtype, sent_emb.dtype, words_embs.dtype, attr_embs.dtype, mask.dtype, cap_lens.dtype)
        # print(self.noise.device, sent_emb.device, words_embs.device, attr_embs.device, mask.device, cap_lens.device)
        # print(self.noise.size(), sent_emb.size(), words_embs.size(), attr_embs.size(), mask.size(), cap_lens.size())
        fake_img, _, _, _ = self.netG(self.noise.to(self.device), sent_emb, words_embs, attr_embs, mask, cap_lens)
        
        im = fake_img[0][0].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
            
        # im.save(os.path.join(save_dir, 'output.png'))
        im.save(save_dir)
    
    def predict(self, prompt, out_path):
        # 1. Pre-process prompt
        text_converter = TextEncoder(train_word_to_idx= self.wordx_to_idx)
        captions, len_caption, attributes = text_converter.process_prompt(prompt)
        # 2. Generate image
        with torch.inference_mode():
            self.generate_img(captions, len_caption, attributes, save_dir = out_path)

    

def main(prompt: str) -> None:
    # 1. set parameters
    cfg_path = 'bird_DAEGAN.yml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. create & load model
    model = DAE_GAN(cfg_file=cfg_path, device=device)
    model.load_models()

    # 3. generate image from prompt
    model.predict(prompt=prompt, out_path='output')


if __name__=='__main__':
    prompt = 'a small red and white bird with a small curved beak'
    main(prompt)