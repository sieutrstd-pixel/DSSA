import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    StridedRandomSliceSequence,
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    YTVIS,
    ClPadToMax1,
    DefaultCollate,
    Xywh2Ltrb,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    CrossEntropyLoss,
    L1Loss,
    ClassAccuracy,
    TensorSize,
    BoxIoU,
    CbLinearCosine,
    Callback,
    HandleLog,
    SaveModel,
)
from object_centric_bench.model import (
    Sequential,
    Interpolate,
    DINO2ViT,
    Identity,
    MLP,
    NormalShared,
    RSFQTransit,
    ARRandTransformerDecoder,
    LearntPositionalEmbedding,
    Linear,
    LayerNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
    DSSA,
    SlotAttention_CMA,
    ObjDiscovRecogn,
)
from object_centric_bench.util import Compose, ComposeNoStar
from object_centric_bench.util_model import interpolat_argmax_attent

### global

max_num = 6 + 1
resolut0 = [256, 256]
resolut1 = [16, 16]
emb_dim = 256
vfm_dim = 384
ncls = 40 + 1
cbox = 4

total_step = 5000
val_interval = total_step // 40
batch_size_t = 32 // 4
batch_size_v = 1
num_work = 4
lr = 1e-3

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    dict(type=RandomCrop, keys=["video", "segment"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["video", "segment"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
transform_v = [
    dict(type=CenterCrop, keys=["video", "segment"], size=None),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
dataset_t = dict(
    type=YTVIS,
    data_file="ytvis_hq/train.lmdb",
    extra_keys=["segment", "bbox", "clazz"],
    transform0=dict(
        type=StridedRandomSliceSequence, keys=["video", "segment", "clazz"], size=5
    ),
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
    ts=30,
)
dataset_v = dict(
    type=YTVIS,
    data_file="ytvis_hq/val.lmdb",
    extra_keys=["segment", "bbox", "clazz"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = dict(  # (b,t,h,w,s) (b,t,s,c) (b,t,s)
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment", "bbox", "clazz"], dims=[3, 1, 1]),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t

### model

# discovery model, same as in ``dssa_r-ytvis_hq.py`` but with ``emb_dim=256``;
# its weights are loaded from the trained DSSA checkpoint via ``ckpt_map``
discov = dict(
    type=DSSA,
    encode_backbone=dict(
        type=Sequential,
        modules=[
            dict(type=Interpolate, scale_factor=0.875, interp="bicubic"),
            dict(
                type=DINO2ViT,
                model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                in_size=int(resolut0[0] * 0.875),
                rearrange=True,
                norm_out=False,
            ),
        ],
    ),
    encode_posit_embed=dict(type=Identity),
    encode_project=dict(
        type=MLP, in_dim=vfm_dim, dims=[vfm_dim, vfm_dim], ln="pre", dropout=0.0
    ),
    initializ=dict(type=NormalShared, num=max_num, dim=emb_dim),
    aggregat=dict(
        type=SlotAttention_CMA,
        num_iter=3,
        embed_dim=emb_dim,
        ffn_dim=emb_dim * 4,
        dropout=0,
        kv_dim=vfm_dim,
        a0_alpha=0.5,
    ),
    transit=dict(
        type=RSFQTransit,
        dt=5,
        ci=vfm_dim,
        c=emb_dim,
        nhead=4,
        expanz=4,
        pdo=0.5,
        norm_first=False,
        bias=False,
    ),
    decode=dict(
        type=ARRandTransformerDecoder,
        emb_dim=vfm_dim,
        posit_embed=dict(
            type=LearntPositionalEmbedding,
            resolut=[resolut1[0] * resolut1[1]],
            embed_dim=vfm_dim,
        ),
        project1=dict(
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=vfm_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        project2=dict(
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=emb_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        backbone=dict(
            type=TransformerDecoder,
            decoder_layer=dict(
                type=TransformerDecoderLayer,
                d_model=vfm_dim,
                nhead=4,
                dim_feedforward=vfm_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            ),
            num_layers=4,
        ),
        readout=dict(type=Identity),
    )
)

recogn = dict(
    type=MLP, in_dim=emb_dim, dims=[emb_dim * 2, ncls + cbox], ln="pre", dropout=0.1
)

model = dict(
    type=ObjDiscovRecogn,
    discov=discov,
    recogn=recogn,
    slotz_idx=1,
    attpd_idx=3,
    segpd_func=lambda _: ptnf.one_hot(
        interpolat_argmax_attent(_.detach(), size=resolut0).long()
    ).bool(),
    slotz_rearr="b t s c -> (b t) s c",
    segpd_rearr="b t h w s -> (b t) (h w) s",
    seggt_rearr="b t h w s -> (b t) (h w) s",
    clsgt_rearr="b t s -> (b t) s",
    boxgt_rearr="b t s c -> (b t) s c",
    ncls=ncls,
    cbox=cbox,
    thresh_iou=1e-1,
)
model_imap = dict(
    input="batch.video",
    seggt="batch.segment",
    clsgt="batch.clazz",
    boxgt="batch.bbox",
)
model_omap = ["slotz", "clspd", "boxpd", "clsgt", "boxgt", "rcidx"]
ckpt_map = [  # target<-source
    ["m.discov.", "m."],  # load trained OCL weights into discov
]
freez = [r"^m\.discov\..*"]

### learn

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = None

loss_fn_t = loss_fn_v = dict(
    ce=dict(
        metric=dict(type=CrossEntropyLoss),
        map=dict(input="output.clspd", target="output.clsgt"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    l1=dict(
        metric=dict(type=L1Loss),
        map=dict(input="output.boxpd", target="output.boxgt"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
        weight=1,
    ),
)
acc_fn_t = dict(
    top1=dict(
        metric=dict(type=ClassAccuracy, topk=1),
        map=dict(input="output.clspd", target="output.clsgt"),  # (?,ncls)->(?,)
    ),
    top3=dict(
        metric=dict(type=ClassAccuracy, topk=3),
        map=dict(input="output.clspd", target="output.clsgt"),  # (?,ncls)->(?,)
    ),
    iou=dict(
        metric=dict(type=BoxIoU),
        map=dict(input="output.boxpd", target="output.boxgt"),  # (?,cbox)
        transforms=dict(type=Xywh2Ltrb, keys=["input"]),
    ),
    num=dict(  # count the number of matched objects
        metric=dict(type=TensorSize, dim=0),
        map=dict(input="output.clsgt"),
    ),
)
acc_fn_v = acc_fn_t.copy()

before_step = [
    dict(
        type=Lambda,
        ikeys=[["batch.video", "batch.segment", "batch.bbox", "batch.clazz"]],
        func=lambda _: _.cuda(),
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step),
    dict(
        type=HandleLog,
        log_file=...,
        ikeys=[["loss.ce", "loss.l1", "acc.top1", "acc.top3", "acc.iou"], ["acc.num"]],
        okeys=[["ce", "l1", "top1", "top3", "iou"], ["num"]],
        ops=["mean", "sum"],
    ),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1]),
    callback_t[1],
    dict(type=SaveModel, save_dir=..., since_step=total_step * 0.5),
]
