from einops import rearrange
import torch as pt
import torch.nn as nn

from .basic import MLP


class DSSA(nn.Module):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        transit,
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode
        self.transit = transit
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    nn.init.orthogonal_(m.weight_ih)
                    nn.init.orthogonal_(m.weight_hh)
                    if m.bias_ih is not None:
                        nn.init.zeros_(m.bias_ih)
                    if m.bias_hh is not None:
                        nn.init.zeros_(m.bias_hh)

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,n,c)
        """
        b, t, c0, h0, w0 = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        slotz = None  # identity slots
        localz = None  # local slots
        attenta = []

        for i in range(t):
            if i == 0:  # (b,n,c)
                query_i = self.initializ(b if condit is None else condit[:, 0, :, :])
                prelocal = None
            else:  # localz: [0,i); encode: [0,i]
                query_i = slotz_i
                prelocal = self.transit(localz, encode[:, : i + 1, :, :])

            niter = None if i == 0 else 1
            (slotz_i, localz_i), attenta_i = self.aggregat(
                encode[:, i, :, :], query_i, prelocal, num_iter=niter
            )

            slotz = (  # (b,i+1,n,c)
                slotz_i[:, None, :, :]
                if slotz is None
                else pt.concat([slotz, slotz_i[:, None, :, :]], 1)
            )
            localz = (  # (b,i+1,n,c)
                localz_i[:, None, :, :]
                if localz is None
                else pt.concat([localz, localz_i[:, None, :, :]], 1)
            )
            attenta.append(attenta_i)  # t*(b,n,h*w)

        attenta = pt.stack(attenta, 1)  # (b,t,n,h*w)
        attenta = rearrange(attenta, "b t n (h w) -> b t n h w", h=h)

        clue = rearrange(feature, "b t c h w -> (b t) (h w) c")
        recon, attentd = self.decode(clue, localz.flatten(0, 1))  # (b*t,h*w,c)
        recon_id, attentd_id = self.decode(clue, slotz.flatten(0, 1))
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        recon_id = rearrange(recon_id, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attentd = rearrange(attentd, "(b t) n (h w) -> b t n h w", b=b, h=h)
        attentd_id = rearrange(attentd_id, "(b t) n (h w) -> b t n h w", b=b, h=h)

        return feature, slotz, localz, attenta, recon, attentd, recon_id, attentd_id


class SlotAttention_CMA(nn.Module):

    def __init__(
        self, num_iter, embed_dim, ffn_dim, dropout=0, kv_dim=None, a0_alpha=1.0
    ):
        """
        - dropout: only works in self.ffn; a bit is beneficial
        - a0_alpha: strength of token ownership, i.e., ``a * a0 ** a0_alpha``;
            1 = strong ownership; 0.5 = balanced; 0 = standard slot attention
        """
        super().__init__()
        kv_dim = kv_dim or embed_dim
        self.num_iter = num_iter
        self.a0_alpha = a0_alpha
        self.norm1q = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1kv = nn.LayerNorm(kv_dim)
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False)
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], None, dropout)
        self.rnn_id = nn.GRUCell(embed_dim, embed_dim)

    def forward(self, input, query, transit, smask=None, num_iter=None):
        """
        - input: (b,h*w,c)
        - query: identity slots, (b,n,c)
        - transit: local slot transition, (b,n,c) or None
        - smask: (b,n), True means valid slot
        """
        b, n, c = query.shape
        query = query if transit is None else query + transit
        local, attent = self.forward_local(input, query, smask, num_iter)
        identity = self.rnn_id(
            local.detach().flatten(0, 1), query.flatten(0, 1)
        ).view(b, n, -1)
        return (identity, local), attent

    def forward_local(self, input, query, smask=None, num_iter=None):
        b, n, c = query.shape
        num_iter = num_iter or self.num_iter

        kv = self.norm1kv(input)
        k = self.proj_k(kv)
        v = self.proj_v(kv)

        q = query
        for _ in range(num_iter):
            x = q
            q = self.norm1q(q)
            q = self.proj_q(q)
            u, a = __class__.inverted_scaled_dot_product_attention(
                q, k, v, smask, a0_alpha=self.a0_alpha
            )
            y = self.rnn(u.flatten(0, 1), x.flatten(0, 1)).view(b, n, -1)
            z = self.norm2(y)
            q = y + self.ffn(z)
        return q, a

    @staticmethod
    def inverted_scaled_dot_product_attention(
        q, k, v, smask=None, eps=1e-5, a0_alpha=1.0
    ):
        scale = q.size(2) ** -0.5
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        if smask is not None:
            logit = logit.where(smask[:, :, None], -pt.inf)
        a0 = logit.softmax(1)  # inverted softmax over queries
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over keys
        if a0_alpha > 0:  # token ownership
            a = a * a0.pow(a0_alpha)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        return o, a0
