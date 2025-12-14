# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import numpy as np
import torch
from einops import rearrange
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_1d(x, freqs):
    """
    x: [B, S, H, C]
    freqs: [S, C//2]  # complex
    """
    B, S, H, C = x.shape
    freqs = freqs[:S, :]
    assert C % 2 == 0
    x_ = x.float().reshape(B, S, H, C // 2, 2)
    x_complex = torch.view_as_complex(x_)  
    x_out = x_complex * freqs[None, :, None, :]
    x_out = torch.view_as_real(x_out).reshape(B, S, H, C)
    return x_out.type_as(x)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_1d_a2v(x, freqs):
    B, S, H, C = x.shape
    c = C//2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    x_i = torch.view_as_complex(x.to(torch.float64).reshape(B, S, H, -1, 2))
    freqs_i = torch.cat([
            freqs[0][:S].view(S, 1, 1, -1).expand(S, 1, 1, -1),
            freqs[1][:1].view(1, 1, 1, -1).expand(S, 1, 1, -1),
            freqs[2][:1].view(1, 1, 1, -1).expand(S, 1, 1, -1)
        ], dim=-1).reshape(1, S, 1, -1)
    x_i = torch.view_as_real(x_i * freqs_i).flatten(3)

    return x_i.to(x.dtype)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_a2v(x, grid_sizes, freqs):
    b, n, c = x.size(0), x.size(2), x.size(3) // 2

    grid_sizes_batches = torch.stack([grid_sizes[0] for i in range(x.size(0))], dim=0)
    f, h, w = grid_sizes_batches[0]
    seq_len = f * h * w
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(1, seq_len, 1, -1)
    x_i = torch.view_as_complex(x.to(torch.float64).reshape(
            b, seq_len, n, -1, 2))
    x_i = torch.view_as_real(x_i * freqs_i).flatten(3)

    return x_i.to(x.dtype)


def reshape_audio_tokens_crrsp_video_frames(audio_tokens, video_len):
    """
    Args:
        audio_tokens(Tensor): Shape [B, L1, C]
        video_len(Int)
    """
    audio_len = audio_tokens.size(1)
    num_video_frames = (video_len - 1) * 4 + 1   # 实际的好视频帧数量
    num_audio_tok_to_video_tok = int((audio_len / num_video_frames) * 4)
    if num_audio_tok_to_video_tok % 2 == 0:
        num_audio_tok_to_video_tok += 1
    pad_audio_len = num_audio_tok_to_video_tok * video_len - audio_len
    pad_audio_tokens = audio_tokens[:,:1].repeat(1,pad_audio_len,1)
    pad_audio_tokens = torch.cat([pad_audio_tokens, audio_tokens], dim=1)

    num_audio_tok_to_video_tok = pad_audio_tokens.size(1) // video_len
    if num_audio_tok_to_video_tok * video_len < pad_audio_tokens.size(1):
        crop_len = num_audio_tok_to_video_tok * video_len - pad_audio_tokens.size(1)
        pad_audio_tokens = pad_audio_tokens[:,crop_len:]
    elif num_audio_tok_to_video_tok * video_len > pad_audio_tokens.size(1):
        pad_len = num_audio_tok_to_video_tok * video_len - pad_audio_tokens.size(1)
        pad_tok = pad_audio_tokens[:,:1].repeat(1, pad_len, 1)
        pad_audio_tokens = torch.cat([pad_tok, pad_audio_tokens], dim=1)

    audio_tokens_reshape = rearrange(pad_audio_tokens, "b (f n) c -> (b f) n c", f=video_len)
    return audio_tokens_reshape


def center_pad_1d(tensor, target_len):
    curr_len = len(tensor)
    left = (target_len - curr_len) // 2
    right = target_len - curr_len - left
    return torch.cat([tensor[0].repeat(left), tensor, tensor[-1].repeat(right)]).long()


def align_audio_to_video(audio_tokens, video_len, window_size):
    """Reshape the input audio sequence to match the video frames"""
    # Calc audio-to-video scale
    valid_video_len = (video_len - 1) * 4 + 1
    sample_rate = audio_tokens.size(1) / valid_video_len
    
    # Calc start and end tokens of each video token
    if window_size % 2 == 0:
        window_size += 1
        
    half_window_size = int(window_size // 2)
    
    indices = [[0 - math.floor(half_window_size * sample_rate), 1 + math.ceil(half_window_size * sample_rate)]]

    max_len = 0
    for i in range(1, video_len):
        start = (i - 1) * 4 + 1
        end = start + 4
        start_b = math.floor(start * sample_rate) - math.floor(half_window_size * sample_rate)
        end_b = math.ceil(end * sample_rate) + math.ceil(half_window_size * sample_rate)
        indices.append([start_b, end_b])
        max_len = max(max_len, end_b - start_b)
    
    def clip(a, lower, upper):
        if a < lower:
            a = lower
        if a >= upper:
            a = upper - 1
        return a
    
    audio_windows = []
    for inds in indices:
        lens = inds[1] - inds[0]  
        
        start, end = inds
        if start < 0:
            left_pad_len = max_len - lens
            start -= left_pad_len
        elif end >= audio_tokens.size(1):
            right_pad_len = max_len - lens
            end += right_pad_len
        else:
            pad_len = max_len - lens
            left_pad_len = pad_len // 2
            right_pad_len = pad_len - left_pad_len
            start -= left_pad_len
            end += right_pad_len
            
        indices = [clip(x, 0, audio_tokens.size(1)) for x in range(start, end)]
        tokens = audio_tokens[:,indices]
        
        audio_windows.append(tokens)
                
    audio_windows = torch.stack(audio_windows, dim=1)

    return audio_windows


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # print(x.dtype)
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs) if grid_sizes is not None else rope_apply_1d(q, freqs),
            k=rope_apply(k, grid_sizes, freqs) if grid_sizes is not None else rope_apply_1d(k, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)


    def forward_audio(self, 
                     x, 
                     context, 
                     context_lens, 
                     freqs_q=None, 
                     freqs_k=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if freqs_q is not None:
            q = rope_apply_1d(q, freqs_q)
            k = rope_apply_1d(k, freqs_k)

        x = flash_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        x = self.o(x)
        return x

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAVJointCrossAttention(nn.Module):   
    def __init__(
        self, 
        q_dim, 
        kv_dim, 
        dim, 
        num_heads, 
        qk_norm=True,
        eps=1e-6, 
        fusion_type="video_to_audio",
    ):
        super().__init__()
        assert fusion_type in ["video_to_audio", "audio_to_video"]
        self.fusion_type = fusion_type
        self.q = nn.Linear(q_dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, q_dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
        if fusion_type == "audio_to_video":
            self.mask_head = Head_mask(q_dim, 1)
        
        nn.init.zeros_(self.o.weight)
        if self.o.bias is not None:
            nn.init.zeros_(self.o.bias)
        
    def forward(self, q_hidden_states, 
                kv_hidden_states, 
                time_embedding, 
                kv_lens, 
                grid_sizes, 
                q_freqs=None, kv_freqs=None, 
                q_3d=True, kv_3d=False,face_mask=None):
        if self.fusion_type == "video_to_audio":
            return self.forward_video_to_audio(q_hidden_states, 
                                               kv_hidden_states, 
                                               kv_lens=kv_lens, 
                                               grid_sizes=grid_sizes, 
                                               q_freqs=q_freqs, 
                                               kv_freqs=kv_freqs, 
                                               q_3d=q_3d, 
                                               kv_3d=kv_3d,
                                               face_mask=face_mask)
        elif self.fusion_type == "audio_to_video":
            return self.forward_audio_to_video(q_hidden_states, 
                                               kv_hidden_states, 
                                               time_embedding=time_embedding, 
                                               grid_sizes=grid_sizes, 
                                               q_freqs=q_freqs, 
                                               kv_freqs=kv_freqs)
            
    def forward_audio_to_video(self, q_hidden_states, 
                               kv_hidden_states, 
                               time_embedding, 
                               grid_sizes, 
                               q_freqs=None, kv_freqs=None):
        """
        Args:
            q_hidden_states(Tensor): Shape [B, L1, C], this is the hidden states of video branch
            kv_hidden_states(Tensor): Shape [B, L2, C], this is the hidden states of audio branch
        """
        b, s, n, d = *q_hidden_states.shape[:2], self.num_heads, self.head_dim
        t, h, w = grid_sizes[0,0].item(), grid_sizes[0,1].item(), grid_sizes[0,2].item()
        
        # predict the face mask region
        self.face_mask_heatmap = self.mask_head(q_hidden_states, time_embedding) 
        self.face_mask_heatmap = self.face_mask_heatmap.to(dtype=q_hidden_states.dtype)
        self.face_mask_heatmap = rearrange(self.face_mask_heatmap, "b (t h w) c -> (b t) (h w ) c", t=t, h=h, w=w)
        
        # reshape the q_hidden_states to frame-level format
        q_hidden_states_reshape = rearrange(q_hidden_states, "b (t h w) c -> (b t) (h w) c", t=t, h=h, w=w)
        
        # reshape the kv_hidden_states to frame-level format according to q_hidden_states_reshape
        kv_hidden_states_reshape = align_audio_to_video(kv_hidden_states, video_len=t, window_size=5)
        
        # compute query, key, value
        q = self.norm_q(self.q(q_hidden_states_reshape)).view(b*t, -1, n, d)
        k = self.norm_k(self.k(kv_hidden_states_reshape)).view(b*t, -1, n, d)
        v = self.v(kv_hidden_states_reshape).view(b*t, -1, n, d)

        # Apply rope
        if q_freqs is not None:
            grid_sizes_copy = grid_sizes.clone()
            grid_sizes_copy[0][0] = 1
            q = rope_apply_a2v(q, grid_sizes_copy, q_freqs)
        
        if kv_freqs is not None:
            k = rope_apply_1d_a2v(k, kv_freqs)
        
        o_hidden_states = flash_attention(q, k, v, k_lens=None)
        o_hidden_states = self.o(o_hidden_states.flatten(2) * self.face_mask_heatmap)
        o_hidden_states = rearrange(o_hidden_states, "(b t) (h w) c -> b (t h w) c", t=t, h=h, w=w)
        return o_hidden_states
    
    def forward_video_to_audio(self, q_hidden_states, kv_hidden_states, 
                               kv_lens, grid_sizes, 
                               q_freqs=None, kv_freqs=None, 
                               q_3d=True, kv_3d=False, face_mask=None):
        """
        Args:
            q_hidden_states(Tensor): Shape [B, L1, C], this is the hidden states of audio branch
            kv_hidden_states(Tensor): Shape [B, L2, C], this is the hidden states of video branch
        """
        
        b, s, n, d = *q_hidden_states.shape[:2], self.num_heads, self.head_dim
        t, h, w = grid_sizes[0,0].item(), grid_sizes[0,1].item(), grid_sizes[0,2].item()

        # Use the predicted face mask to guide the v2a genetation
        face_mask_heatmap = face_mask         

        # reshape the q_hidden_states (audio) to frame-level format
        q_hidden_states_reshape = q_hidden_states
        fl = 1+(t-1)*4
        sample_rate = s / fl
        ind_a2v = torch.arange(0, s).to(q_hidden_states_reshape) / sample_rate
        ind_a2v_down = ind_a2v.floor().clamp(0, fl-1)
        ind_a2v_up = (ind_a2v_down + 1).clamp(0, fl-1) 

        kv_reshape = rearrange(kv_hidden_states, "b (t h w) c -> (b t) (h w) c", t=t, h=h, w=w)
        kv_reshape = kv_reshape * face_mask_heatmap
        kv_reshape_back = rearrange(kv_reshape, "(b t) (h w) c -> b t (h w) c", t=t, h=h, w=w)
        kv_reshape_back = torch.cat([kv_reshape_back[:, :1], kv_reshape_back[:, 1:].repeat_interleave(4, dim=1)], 1)
        
        kv_hidden_states_reshape = (1-(ind_a2v_up-ind_a2v))[None, :, None, None] * kv_reshape_back[:, ind_a2v_up.long()] + \
                (1-(ind_a2v-ind_a2v_down))[None, :, None, None] * kv_reshape_back[:, ind_a2v_down.long()]

        # compute query, key, value
        q = self.norm_q(self.q(q_hidden_states_reshape)).reshape(b*s, -1, n, d)
        k = self.norm_k(self.k(kv_hidden_states_reshape)).reshape(b*s, -1, n, d)
        v = self.v(kv_hidden_states_reshape).reshape(b*s, -1, n, d)
        
        # Apply rope
        if q_freqs is not None:
            if q_3d:
                grid_sizes_copy = grid_sizes.clone()
                grid_sizes_copy[0][0] = 1
                q = rope_apply_a2v(q, grid_sizes_copy, q_freqs)
            else:
                q = rope_apply_1d_a2v(q, q_freqs)
        if kv_freqs is not None:
            if kv_3d:
                grid_sizes_copy = grid_sizes.clone()
                grid_sizes_copy[0][0] = 1
                k = rope_apply_a2v(k, grid_sizes_copy, kv_freqs)
            else:
                k = rope_apply_1d_a2v(k, kv_freqs)
        
        o_hidden_states = flash_attention(q, k, v, k_lens=kv_lens)
        o_hidden_states = self.o(o_hidden_states.flatten(2))

        o_hidden_states = rearrange(o_hidden_states, "f b c -> b f c")

        return o_hidden_states


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6 
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim,
                                            num_heads,
                                            (-1, -1),
                                            qk_norm,
                                            eps
                                            )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context_text,           # Either TTS or VideoCaption, depands on whether this is called in video-branch or audio-branch
            context_lens,    
            context_text_lens,       
            branch="video",
    ):
        
        assert branch in ["audio", "video"]
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        e_2 = [e[i].to(dtype=x.dtype, device=x.device) for i in range(6)]
        xx = self.norm1(x)
        xx = xx * (1 + e_2[1].squeeze(2)) + e_2[0].squeeze(2)

        y = self.self_attn(xx, context_lens, grid_sizes, freqs)
        x = x + y * e_2[2].squeeze(2)
        x_norm = self.norm3(x)

        if branch == "audio":
            x = x + self.cross_attn.forward_audio(x_norm, context_text, context_text_lens, freqs, freqs)
        else:
            x = x + self.cross_attn(x_norm, context_text, context_lens)

        y = self.ffn(self.norm2(x) * (1 + e_2[4].squeeze(2)) + e_2[3].squeeze(2))

        with torch.amp.autocast('cuda', dtype=e_2[0].dtype):
            x = x + y * e_2[5].squeeze(2)   
        return x


class Head_audio(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class Head_mask(nn.Module):
    def __init__(self, dim, out_dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.eps = eps

        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)
        
        # act
        self.act = nn.Sigmoid()
    
    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
            x = self.act(x)
        return x
 

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x
    
    
class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=4, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = torch.nn.functional.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text
    

class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """
    @register_to_config
    def __init__(self,
                 model_type='ti2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 in_audio_dim=80,
                 dim=3072,
                 ffn_dim=14336,
                 audio_dim=1536,
                 audio_ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 audio_cond_dim=512,
                 out_dim=48,
                 out_dim_audio=80,
                 num_heads=24,
                 audio_num_heads=12,
                 num_layers=30,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        
        assert model_type in ['t2v', 'i2v', 'flf2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.audio_dim = audio_dim
        self.ffn_dim = ffn_dim
        self.audio_ffn_dim = audio_ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.audio_num_heads = audio_num_heads
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings - video or image
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # embeddings - audio
        self.audio_embedding = nn.Sequential(
            nn.Linear(in_audio_dim, audio_dim), nn.GELU(approximate='tanh'),
            nn.Linear(audio_dim, audio_dim))    
        
        self.audio_padding = nn.Parameter(torch.randn(audio_dim), requires_grad=True) 

        self.text_embedding_video = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))    
        self.text_embedding = TextEmbedding(
            text_num_embeds=2545, text_dim=512, mask_padding=True, conv_layers=4, conv_mult=2)
        self.text_proj = nn.Linear(512, audio_dim, bias=False)    

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.time_embedding_audio = nn.Sequential(
            nn.Linear(freq_dim, audio_dim), nn.SiLU(), nn.Linear(audio_dim, audio_dim))
        self.time_projection_audio = nn.Sequential(nn.SiLU(), nn.Linear(audio_dim, audio_dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)    
            for _ in range(num_layers)
        ])
        self.audio_blocks = nn.ModuleList([
            WanAttentionBlock(audio_dim, audio_ffn_dim, audio_num_heads,
                              window_size, qk_norm, cross_attn_norm, eps) 
            for _ in range(num_layers)
        ])
        self.a2v_blocks_mapping = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]
        self.a2v_blocks = nn.ModuleList([
            WanAVJointCrossAttention(q_dim=dim, kv_dim=audio_dim, dim=dim, 
                                     num_heads=num_heads, qk_norm=qk_norm, 
                                     fusion_type="audio_to_video") 
            for _ in range(len(self.a2v_blocks_mapping))
        ])
        
        self.v2a_blocks_mapping = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]
        self.v2a_blocks = nn.ModuleList([
            WanAVJointCrossAttention(q_dim=audio_dim, kv_dim=dim, dim=audio_dim, 
                                     num_heads=audio_num_heads, qk_norm=qk_norm, 
                                     fusion_type="video_to_audio") 
            for _ in range(len(self.v2a_blocks_mapping))
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        self.head_audio = Head_audio(audio_dim, out_dim_audio, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(8192, d - 4 * (d // 6)),
            rope_params(8192, 2 * (d // 6)),
            rope_params(8192, 2 * (d // 6))
        ], dim=1)

        self.freqs_audio = rope_params(8192, audio_dim // audio_num_heads)


    def forward(
            self,
            video,              
            audio,              
            t,
            context_list,      
            seq_len,
            video_cond=1,
            audio_cond=0,
            zero_con=False,
            no_v2a=False,
            no_a2v=False,
            slg=-1
    ):
        if zero_con:
            no_v2a = True
            no_a2v = True

        device = self.text_embedding.text_embed.weight.device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if self.freqs_audio.device != device:
            self.freqs_audio = self.freqs_audio.to(device)

        (
            context_text, ref_audio, cond_audio,       # audio
            ref_image, cond_video, context_text_caption,  # video
        ) = (
            context_list[0], context_list[1], context_list[2],
            context_list[3], context_list[4], context_list[5]
        )

        # ------------------------- video
        if cond_video is not None:
            cat_video = [torch.cat([v, u[:, v.shape[1]:]], dim=1) for u, v in zip(video, cond_video)]   
        if ref_image is not None:
            cat_video = [torch.cat([u, v], dim=1) for u, v in zip(ref_image, cat_video)]
        cat_video = [self.patch_embedding(u.unsqueeze(0)) for u in cat_video]
        frame_l = cat_video[0].shape[-1] * cat_video[0].shape[-2]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in cat_video])  
        grid_sizes_2 = torch.stack(
            [torch.tensor((u.shape[2]-1, u.shape[3], u.shape[4]), dtype=torch.long) for u in cat_video])  
        cat_video = [u.flatten(2).transpose(1, 2) for u in cat_video] 
        seq_lens_video = torch.tensor([u.size(1) for u in cat_video], dtype=torch.long)
        seq_len2 = seq_lens_video.item() 
        if seq_len2 != 0:
            cat_video = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len2 - u.size(1), u.size(2))],
                          dim=1) for u in cat_video
            ])
        # --- encode text embedding for video generation
        context_video_lens = None
        context_video = self.text_embedding_video(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context_text_caption
            ]))


        # ------------------------- audio 
        if ref_audio[0] == None:
            ref_audio = [a[:0] for a in audio]
        if cond_audio[0] == None:
            cond_audio = [a[:0] for a in audio]
        max_len = max([a.size(0)+b.size(0) for a, b in zip(audio, ref_audio)])
        targ_lens = [a.size(0) for a in audio]                  
        ref_audio_lens = [a.size(0) for a in ref_audio]         
        context_audio_lens = [r.size(0)+a.size(0) for r, a in zip(ref_audio, audio)]
        context_audio_lens = torch.from_numpy(np.array(context_audio_lens)).long().to(device=device)
        cat_audio = []
        
        # --- encode audio embeddings
        for ra, na, cona in zip(ref_audio, audio, cond_audio):
            pad_len = max_len - (na.size(0) + ra.size(0))
            ca = torch.cat([ra, cona[:audio_cond], na[audio_cond:]], dim=0).unsqueeze(0)    
            ca = self.audio_embedding(ca)
            if pad_len > 0:
                pa = self.audio_padding.view(1,1,-1).repeat(1,pad_len,1).to(dtype=ca.dtype)
                ca = torch.cat([ca, pa], dim=1)    
            cat_audio.append(ca)
        cat_audio = torch.cat(cat_audio, dim=0)
        seq_lens_audio = torch.tensor([cat_audio.size(1)], dtype=torch.long).repeat(cat_audio.size(0)) 
        # --- encode text embedding for audio generation
        context_speech_lens = seq_len
        cat_speech_text = self.text_embedding(context_text, seq_len=context_text.size(1))
        cat_speech_text = self.text_proj(cat_speech_text) 
        

        # ------------------------- time embeddings
        la = 1
        if t.dim() == 1:
            tv = t.view(-1,1).repeat(t.size(0), seq_len2)   # (b f)
            tv[:, :frame_l*(1+video_cond)] = 0.
            ta = t.view(-1,1).repeat(t.size(0), la)         # (b 1)
            ta[:, :audio_cond] = 0.
            
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            tv = tv.flatten()
            ta = ta.flatten()

            e_video = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, tv).unflatten(0, (bt, seq_len2)).float())    # (b f c)
            e0_video = self.time_projection(e_video).unflatten(2, (6, self.dim))                    # (b f 6 c)
            
            e_audio = self.time_embedding_audio(
                sinusoidal_embedding_1d(self.freq_dim, ta).unflatten(0, (bt, la)).float())          # (b 1 c)
            e0_audio = self.time_projection_audio(e_audio).unflatten(2, (6, self.audio_dim))        # (b 1 6 c)
        
        # ------------------------- layer forward
        
        for idx, (audio_block, video_block) in enumerate(zip(self.audio_blocks, self.blocks)):
            if slg > 0 and idx == slg: 
                continue

            audio_kwargs = dict(
                e=e0_audio,
                seq_lens=seq_lens_audio,
                grid_sizes=None,
                freqs=self.freqs_audio,
                context_text=cat_speech_text,
                context_lens=context_audio_lens,
                context_text_lens=context_speech_lens,
                branch="audio",
            )

            video_kwargs = dict(
                e=e0_video,
                seq_lens=seq_lens_video,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context_text=context_video,
                context_lens=context_video_lens,
                context_text_lens=None,
                branch="video",
            )
            
            # --- audio block
            cat_audio = audio_block(cat_audio, **audio_kwargs)
            # --- video block
            cat_video = video_block(cat_video, **video_kwargs)

            # ----- split target audio and ref audio -----
            if idx in self.a2v_blocks_mapping or idx in self.v2a_blocks_mapping:
                split_ref_audio, split_audio = [], []
                for rl, ca in zip(ref_audio_lens, cat_audio):
                    split_ref_audio.append(ca[:rl])
                    split_audio.append(ca[rl:])
                split_ref_audio = torch.stack(split_ref_audio, dim=0)   
                split_audio = torch.stack(split_audio, dim=0)           
            
            # --- Audio to Video
            if idx in self.a2v_blocks_mapping:
                a2v_index = self.a2v_blocks_mapping.index(idx)
                a2v_block = self.a2v_blocks[a2v_index]
                
                fused_video = a2v_block(cat_video[:, frame_l:], split_audio, e_video[:, frame_l:],
                                                context_audio_lens, grid_sizes_2, 
                                                self.freqs, self.freqs_audio, 
                                                True, False)
                                
                # Get face_mask of current audio-to-video block
                face_mask_heatmap = a2v_block.face_mask_heatmap
                self.grid_sizes = grid_sizes
                
            # --- Video to Audio
            if idx in self.v2a_blocks_mapping:
                v2a_index = self.v2a_blocks_mapping.index(idx)
                v2a_block = self.v2a_blocks[v2a_index]
                
                fused_audio = v2a_block(split_audio, cat_video[:, frame_l:], None,
                                                context_video_lens, grid_sizes_2, 
                                                self.freqs_audio, self.freqs, 
                                                False, True, face_mask_heatmap)
            
            if idx in self.a2v_blocks_mapping:
                cat_part = cat_video[:, frame_l*(1+video_cond):]
                fused_part = fused_video[:, frame_l*video_cond:]

                new_part = cat_part + (1 - int(no_a2v)) * fused_part

                cat_video = torch.cat([
                    cat_video[:, :frame_l*(1+video_cond)], 
                    new_part 
                ], dim=1)

            if idx in self.v2a_blocks_mapping:
                start_idx = ref_audio_lens[0] + audio_cond

                cat_audio_front = cat_audio[:, :start_idx]

                cat_audio_back = cat_audio[:, start_idx:]
                fused_audio_part = fused_audio[:, audio_cond:]
                new_back = cat_audio_back + (1 - int(no_v2a)) * fused_audio_part

                cat_audio = torch.cat([cat_audio_front, new_back], dim=1)
            
        # head
        out_video = []
        for v, e in zip(cat_video, e_video):
            ov = self.head(v.unsqueeze(0), e.unsqueeze(0))
            ov = self.unpatchify(ov, grid_sizes)[0]
            out_video.append(ov.float())
        
        out_audio = []
        for (a, l, e, rl) in zip(cat_audio, targ_lens, e_audio, ref_audio_lens):
            oa = a[rl:]
            oa = self.head_audio(oa[:l].unsqueeze(0), e.unsqueeze(0))
            out_audio.append(oa)
        
        return out_video, out_audio 

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
