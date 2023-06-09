from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg
import nltk
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

nltk.data.path.append("Your nltk package path")

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys, attrs = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))
    captions = captions[sorted_cap_indices].squeeze()
    attrs = attrs[sorted_cap_indices].squeeze()

    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        attrs = Variable(attrs).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        attrs = Variable(attrs)
        sorted_cap_lens = Variable(sorted_cap_lens)
   # print(captions.shape, attrs.shape)
    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, attrs]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
           # print("imsize[i]:", imsize[i], img.size)
          #  if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Resize(imsize[i])(img)
          #  else:
           #      re_img = img
           # print("re_img.shape:", re_img.size)
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            # print("base_size, i:", base_size, i)
            self.imsize.append(base_size)
            if i == 0 or i == (cfg.TREE.BRANCH_NUM-2):
                base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.attrs, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        all_attr = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                #    print(cnt, "cap:", cap)
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # sentence = tokenizer.tokenize(text.lower())

                    # Attribute extraction
                    sentence_tag = nltk.pos_tag(tokens)
                    # CUB
                    grammar = "NP: {<DT>*<JJ>*<CC|IN>*<JJ>+<NN|NNS>+|<DT>*<NN|NNS>+<VBZ>+<JJ>+<IN|CC>*<JJ>*}"
                    # COCO
                  #  grammar = "NP: {<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+|<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+<IN>+<NN|NNS>+|<VB|VBD|VBG|VBN|VBP|VBZ>+<CD|DT>*<JJ|PRP$>*<NN|NNS>+|<IN>+<DT|CD|JJ|PRP$>*<NN|NNS>+<IN>*<CD|DT>*<JJ|PRP$>*<NN|NNS>*}"
                    cp = nltk.RegexpParser(grammar)
                    tree = cp.parse(sentence_tag)
                    
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue                    

                    attr_list = []

                    for i in range(len(tree)):
                        if type(tree[i]) == nltk.tree.Tree:
                            attr = []
                            for j in range(len(tree[i])):
                                attr.append(tree[i][j][0])
                            attr_list.append(attr)
                    # attribute end
                    all_attr.append(attr_list)  # [cnt, att_num, num_words]
                    # print('tokens', tokens)
                  #  if len(tokens) == 0:
                  #      print('cap', cap)
                  #      continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)  # [cnt, num_words]
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
              #  print("len(captions):",len(captions))
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d, len(captions)=%d, embedding_num=%d'
                          % (filenames[i], cnt, len(captions), self.embeddings_num))
        return all_captions, all_attr

    def build_dictionary(self, train_captions, test_captions, train_attrs, test_attrs):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        train_attrs_new = []
        for t in train_attrs:
            rev = []
            for attr in t:
                attr_new = []
                for w in attr:
                    if w in wordtoix:
                        attr_new.append(wordtoix[w])
                rev.append(attr_new)
            train_attrs_new.append(rev)

        test_attrs_new = []
        for t in test_attrs:
            rev = []
            for attr in t:
                attr_new = []
                for w in attr:
                    if w in wordtoix:
                        attr_new.append(wordtoix[w])
                rev.append(attr_new)
            test_attrs_new.append(rev)

        return [train_captions_new, test_captions_new, train_attrs_new, test_attrs_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        
        filepath = os.path.join(data_dir, 'captions.pickle')
        print("filepath:", filepath)
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions, train_attrs = self.load_captions(data_dir, train_names)
            test_captions, test_attrs = self.load_captions(data_dir, test_names)

            train_captions, test_captions, train_attrs, test_attrs, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions, train_attrs, test_attrs)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix, train_attrs, test_attrs], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions, train_attrs, test_attrs = x[0], x[1], x[4], x[5]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            attrs = train_attrs
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            attrs = test_attrs
            filenames = test_names
        return filenames, captions, attrs, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
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
        return x, x_len

    def get_attr(self, sent_ix):
        # sen_attr = np.asarray(self.attrs[sent_ix]).astype('int64')
        sen_attr = self.attrs[sent_ix]
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

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        attrs = self.get_attr(new_sent_ix)
      #  print(caps.shape, attrs.shape)
        return imgs, caps, cap_len, cls_id, key, attrs

    def get_mis_caption(self, cls_id):
        mis_match_captions_t = []
        mis_match_captions = torch.zeros(99, cfg.TEXT.WORDS_NUM)
        mis_match_captions_len = torch.zeros(99)
        i = 0
        while len(mis_match_captions_t) < 99:
            idx = random.randint(0, self.number_example)
            if cls_id == self.class_id[idx]:
                continue
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = idx * self.embeddings_num + sent_ix
            caps_t, cap_len_t = self.get_caption(new_sent_ix)
            mis_match_captions_t.append(torch.from_numpy(caps_t).squeeze())
            mis_match_captions_len[i] = cap_len_t
            i = i +1
        sorted_cap_lens, sorted_cap_indices = torch.sort(mis_match_captions_len, 0, True)
        #import ipdb
        #ipdb.set_trace()
        for i in range(99):
            mis_match_captions[i,:] = mis_match_captions_t[sorted_cap_indices[i]]
        return mis_match_captions.type(torch.LongTensor).cuda(), sorted_cap_lens.type(torch.LongTensor).cuda()

    def __len__(self):
        return len(self.filenames)
