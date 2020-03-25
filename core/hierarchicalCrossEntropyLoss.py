import torch

class CrossEntropyLossoneWithOneHot(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(CrossEntropyLossoneWithOneHot, self).__init__()
        self.target_inv = None
        self.zeros_fill = None
        self.device = device

    def _log_softmax(self, x):
        b = x.max(axis=1).values
        b = b.view(-1, 1)
        b = b.repeat(1, x.shape[1])
        return (x-b) - torch.log(torch.exp(x-b).sum(-1)).unsqueeze(-1)
    
#     def _softmax(self, x):
#         return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

    def _nll(self, input, target, weight):
        # https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss
        
        input = input.masked_scatter(target, self.zeros_fill.float()).to(self.device)
        if weight is not None:
            input = -torch.mul(input, weight)
            weight = weight.masked_scatter(target, self.zeros_fill.float()).to(self.device)
            weight = torch.sum(weight)
            input = torch.div(input, weight)
            input = torch.sum(input)
        else:
            input = torch.sum(input, 1, keepdim=True)
            input = -input.mean()
        return input
    
    def _encode2onehot(self, output, target):
        target_onehot = torch.zeros(output.shape).to(self.device)
        target_onehot.scatter_(1, target.view(-1,1), 1)
        return target_onehot
    
    def forward(self, output, target, weight):
        if target.dim() == 1:
            target_onehot = self._encode2onehot(output, target)
        elif target.dim() == 2:
            target_onehot = target
        self.zeros_fill = torch.zeros_like(target_onehot, dtype=torch.int).to(self.device)
        self.target_inv = torch.eq(target_onehot.int(), self.zeros_fill)

        if weight is not None:
            if weight.shape != self.target_inv.shape:
                weight = weight.repeat(self.target_inv.shape[0]).view(self.target_inv.shape)
        
        loss = self._log_softmax(output)
        loss = self._nll(loss, self.target_inv, weight)
        return loss

class hierarchicalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, coeff, hierarchy_dict, 
                 weight = None, device='cpu'):
        super(hierarchicalCrossEntropyLoss,self).__init__()
        self.coeff = coeff
        self.hierarchy_dict = hierarchy_dict
        self.weight = weight
        if self.weight == None:
            self.weight = [None] * len(self.coeff)
        self.device = device
    
    def label2hierarchical_label(self, labels, level):
        new_labels = []
        for i, l in enumerate(labels):
            for k, v in self.hierarchy_dict[str(level)].items():
                if v[int(l)] == 1:
                    new_l = self.hierarchy_dict[str(level)][k]
                    new_labels.append(new_l)
        return torch.tensor(new_labels).to(self.device)
        
    def forward(self, output, target):
        base_criterion = CrossEntropyLossoneWithOneHot(self.device)
        loss = None
        
        for i, c in enumerate(self.coeff):
            if i == 0:
#                 print('level:{}, c:{}, inputshape:{}, targetshape:{}'.format(i, c, output.shape, target.shape))
                # normal CrossEntropy
                loss = torch.mul(base_criterion(output, target, self.weight[i]), c)
            elif i > 0:
                h_label = self.label2hierarchical_label(target, i)     
#                 print('level:{}, c:{}, inputshape:{}, targetshape:{}'.format(i, c, output.shape, h_label.shape))
                loss = torch.add(loss, torch.mul(base_criterion(output, h_label, self.weight[i]), c))            
#             print('loss={}'.format(loss))

        return loss