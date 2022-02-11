import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data + '_data.pkl' if if_align else data + '_data_noalign.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 3

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'),
                    self.meta[index][2].decode('UTF-8'))
        return X, Y, META


# using bert-text features-:
class MMDataset(Dataset):
    def __init__(self, dataset_path, use_bert=True, need_norm=False, train_mode='classification',
                 data='mosi', split='train', if_align=False):
        super(MMDataset, self).__init__()
        self.split = split
        self.use_bert = use_bert
        self.need_norm = need_norm
        self.if_align = if_align

        if self.if_align:
            self.dataset_path = dataset_path + '/aligned_50.pkl'
        else:
            if data == 'sims':
                self.dataset_path = dataset_path + '/unaligned_39.pkl'
            else:
                self.dataset_path = dataset_path + '/unaligned_50.pkl'
        self.data = data
        self.train_mode = train_mode
        self.n_modalities = 3
        self.seq_lens = (50, 50, 50)
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[data]()

    def __init_mosi(self):
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        if self.use_bert:
            print('using bert')
            self.text = data[self.split]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.split]['text'].astype(np.float32)
        self.vision = data[self.split]['vision'].astype(np.float32)
        self.audio = data[self.split]['audio'].astype(np.float32)
        self.rawText = data[self.split]['raw_text']
        self.ids = data[self.split]['id']

        self.labels = {
            'M': data[self.split][self.train_mode + '_labels'].astype(np.float32)
        }
        if self.data == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.split][self.train_mode + '_labels_' + m]

        if not self.if_align:
            self.audio_lengths = data[self.split]['audio_lengths']
            self.vision_lengths = data[self.split]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.need_norm:
            self.__normalize()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __truncated(self):
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if ((instance[index] == padding).all()):
                        if (index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index + 20])
                            break
                    else:
                        truncated_feature.append(instance[index:index + 20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        text_length, audio_length, video_length = self.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_dim(self):
        if self.use_bert:
            return 768, self.audio.shape[2], self.vision.shape[2]
        else:
            return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_n_modalities(self):
        return self.n_modalities

    def __getitem__(self, index):
        X = (index, torch.Tensor(self.text[index]), torch.Tensor(self.audio[index]), torch.Tensor(self.vision[index]))
        Y = torch.Tensor([self.labels['M'][index].reshape(-1)])
        META = (0, 0, 0)

        return X, Y, META
