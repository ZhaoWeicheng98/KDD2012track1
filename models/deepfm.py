import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    '''
    A DeepFM network with RMSE loss for rates prediction problem.
    '''

    def __init__(self, feature_sizes, embedding_size=4, hidden_dims=[32, 32], num_classes=1, dropout=[0.5, 0.5], use_cuda=True):
        """
        Initialize a new network
        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - use_cuda: Bool, Using cuda or not
        """
        super(DeepFM, self).__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.bias = nn.Parameter(torch.randn(1))

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.fm_embeddings1 = nn.ModuleList(
            [nn.Embedding(feature_size+1, 1) for feature_size in self.feature_sizes])
        self.fm_embeddings2 = nn.ModuleList([nn.Embedding(
            feature_size+1, self.embedding_size) for feature_size in self.feature_sizes])

        all_dims = [self.field_size*self.embedding_size] + \
            self.hidden_dims+[self.num_classes]
        for i in range(1, len(hidden_dims)+1):
            setattr(self, 'linear_'+str(i),
                    nn.Linear(all_dims[i-1], all_dims[i]))
            setattr(self, 'batchNorm_'+str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))

    def forward(self, xi, xv):
        """
        Forward process of network. 
        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        fm_emb_arr1 = [(torch.sum(emb(xi[:, i,:]), 1).t() * xv[:, i]).t()
                       for i, emb in enumerate(self.fm_embeddings1)]
        # print(fm_emb_arr1)
        fm_order1 = torch.cat(fm_emb_arr1, 1)
        # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        fm_emb_arr2 = [(torch.sum(emb(xi[:, i,:]), 1).t() * xv[:, i]).t()
                       for i, emb in enumerate(self.fm_embeddings2)]
        fm_sum_order_emb2 = sum(fm_emb_arr2)
        fm_sum_order_emb2_square = fm_sum_order_emb2 * \
            fm_sum_order_emb2  # (x+y)^2
        fm_order_emb2_square = [
            item*item for item in fm_emb_arr2]
        fm_order_emb2_square_sum = sum(
            fm_order_emb2_square)  # x^2+y^2
        fm_order2 = (fm_sum_order_emb2_square -
                     fm_order_emb2_square_sum) * 0.5

        deep_emb = torch.cat(fm_emb_arr2, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)

        total_sum = torch.sum(fm_order1, 1) + \
            torch.sum(fm_order2, 1) + torch.sum(deep_out, 1) + self.bias
        return total_sum

    def fit(self, loader_train, loader_val, optimizer, epochs=20, print_every=100):
        """
        Training a model and valid accuracy.
        Inputs:
        - loader_train: I
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - print_every: Integer, print after every number of iterations. 
        """
        model = self.train().to(device=self.device)
        criterion = nn.BCEWithLogitsLoss()

        for i in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=torch.long)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)

                total = model(xi, xv)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % print_every == 0:
                    print('Epoch:%d, Iteration %d, loss = %.4f' % (i,t, loss.item()))
                    self.check_accuracy(loader_val, model)
                    print()

            torch.save(model.state_dict(), 'weights/deepFM_{}.pth'.format(i))

    def check_accuracy(self, loader, model):
        """
        Check accurancy.
        """
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                # move to device, e.g. GPU
                xi = xi.to(device=self.device, dtype=torch.long)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.uint8)
                total = model(xi, xv)
                preds = (F.sigmoid(total) > 0.5)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' %
                  (num_correct, num_samples, 100 * acc))
