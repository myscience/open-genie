from genie.tokenizer import VideoTokenizer
from genie.tokenizer import MAGVIT2_ENC_DESC
from genie.tokenizer import MAGVIT2_DEC_DESC

from genie.action import LatentAction
from genie.dynamics import DynamicsModel

from genie.genie import Genie

LATENT_ACT_ENC = (
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 16,
    }),
    ('spacetime_downsample', {
        'in_channels' : 256,
        'kernel_size' : 3,
        'time_factor' : 1,
        'space_factor' : 2,
    }),
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 16,
    }),
)

LATENT_ACT_DEC = (
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 16,
        'has_ext' : True,
        'time_attn_kw'  : {'key_dim' : 8},
    }),
    ('spacetime_upsample', {
        'in_channels' : 256,
        'kernel_size' : 3,
        'time_factor' : 1,
        'space_factor' : 2,
    }),
    ('space-time_attn', {
        'n_rep' : 2,
        'n_embd' : 256,
        'n_head' : 4,
        'd_head' : 16,
        'has_ext' : True,
        'time_attn_kw'  : {'key_dim' : 8},
    }),
)