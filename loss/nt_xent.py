import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine):
        if use_cosine:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        '''
        The output matrix contains positive and negative samples. For extracting
        the negative ones we need to select all the elements excepts those on
        diagonal (similarity of each data with its own), and those on distance of
        batch_size to the diagonal (positive ones).
        '''
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (2N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (2N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        '''
        Change the shape x and y in a way that the last dimension (dim=-1) has
        same number, and the output will be a matrix of size 2N which each row
        has the Cosine similarity between that data and other.
        For example, in the output matrix the row 2 and column 3 has the Cosie
        similarity of data 2 and data 3.
        Note: since the Cosine is symmetric, the matrix will be symmetric too!
        '''
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        # x shape: (2N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (2N, 2N)
        return v

    def forward(self, zis, zjs):
        '''
        First the datas are concatinated for finding the similarity in one line.
        Then positive and negative samples are extracted and concatinated toghether.
        Since the positive ones are in the first column, the output label will be
        all zero (first column). These will be forwarded to CrossEntropy loss
        -log(exp(x[class]) / (\sum_j exp(x[j]))) to compute the final value.
        '''
        # zjs shape: (N, C)
        # zis shape: (N, C)
        # representations shape: (2N, C)
        representations = torch.cat([zjs, zis], dim=0)
        # similarity_matrix shape: (2N, 2N)
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # positives shape: (2N, 1)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # positives shape: (2N, 2(N-1))
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        # logits shape: (2N, 2N-1)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        # logits shape: (2N)
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
