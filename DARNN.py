class DARNN(nn.Module):
    """Implementation of https://arxiv.org/pdf/1704.02971.pdf"""
    def __init__(self, T, n, m, p, out_len):
        super().__init__()
        self.T = T
        self.n = n
        self.m = m
        self.p = p
        self.out_len = out_len
        
        self.emb = nn.Embedding(num_embeddings=500, embedding_dim=m)
        self.We = nn.Linear(2*m, T)
        self.Ue = nn.Linear(T, T)
        self.Ve = nn.Linear(T, 1)
        self.a = nn.Softmax()
        self.encoder_lstm = nn.LSTM(input_size=n, hidden_size=m, num_layers=1, dropout=.7)
        
        self.Wd = nn.Linear(2*p, m)
        self.Ud = nn.Linear(m, m)
        self.Vd = nn.Linear(m, 1)
        self.b = nn.Softmax(dim=0)
        self.y_hat = nn.Linear(1+m, 1)
        self.decoder_lstm = nn.LSTM(input_size=1, hidden_size=p, num_layers=1, dropout=.5)
        self.Wy = nn.Linear(p+m, p)
        self.Vy = nn.Linear(p, 1)
    
    def forward(self, x):
        syms = x[:,0,0].long()
        emb = self.emb(syms).unsqueeze(0) # 1 x bs x m
        
        bs = x.size(0)
        x = x[:,:,1:] # bs x T x n
        T, n, m, p = self.T, self.n, self.m, self.p
        targ_series = x[:,:,0]
        h = V(torch.zeros(1, bs, m)) # 1 x bs x m
        s = emb
        
        encoded = []
        for t in range(T):
            conc_hs = torch.cat((h, s), dim=-1) # 1 x bs x 2m
            # Ue => T x T
            We_applied = self.We(conc_hs).transpose(0,1) # bs x 1 x T
            Ue_applied = self.Ue(x.transpose(1,2)) # bs x n x T
            e = F.relu(self.Ve(We_applied + Ue_applied)).squeeze() # bs x n
            a = self.a(e) # bs x n
            x_hat = (a * x[:,t,:]).unsqueeze(0) # 1 x bs x n
            _, (h, s) = self.encoder_lstm(x_hat, (h, s))
            encoded.append(h)
        encoded = torch.stack(encoded) # T x 1 x bs x m
        encoded = encoded.transpose(1,2).view(T, bs, -1) # T x bs x 1*m
        
        d = V(torch.zeros(1, bs, p)) # 1 x bs x p
        sd = V(torch.zeros(1, bs, p)) # 1 x bs x p
        y = targ_series[:,-1] # bs
        preds = []
        for t in range(self.out_len):
            conc_ds = torch.cat((d, sd), dim=-1) # 1 x bs x 2p
            Wd_applied = self.Wd(conc_ds) # 1 x bs x m
            Ud_applied = self.Ud(encoded) # T x bs x m
            l = self.Vd(Wd_applied + Ud_applied) # T x bs x 1
            b = self.b(l).permute(1,2,0) # bs x 1 x T
            c = (b @ encoded.transpose(0,1)).squeeze() # bs x m
            
            conc_yc = torch.cat((y.unsqueeze(1), c), dim=-1) # bs x 1+m
            y_hat = self.y_hat(conc_yc) # bs x 1
            y_hat = y_hat.unsqueeze(0) # 1 x bs x 1
            _, (d, sd) = self.decoder_lstm(y_hat, (d, sd))
            conc_dc = torch.cat((d.squeeze(), c), dim=-1) # bs x p+m
            pred = self.Wy(conc_dc) # bs x p
            pred = self.Vy(pred).squeeze() # bs
            preds.append(pred)
            y = pred
        preds = torch.stack(preds).transpose(0,1) # bs x out_len
        return preds
