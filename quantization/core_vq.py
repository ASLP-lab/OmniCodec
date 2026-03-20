# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

from einops import rearrange, repeat
from functools import partial, cache
import torch
from torch import nn
from torch import distributed
import torch.nn.functional as F
import einx
from torch import einsum, is_tensor, Tensor
from einops import rearrange, repeat, reduce, pack, unpack
from typing import Callable
from torch.amp import autocast


class _CodebookForwardResult(tp.NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    metrics: tp.Dict[str, torch.Tensor]

class _VQForwardResult(tp.NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    metrics: tp.Dict[str, torch.Tensor]
    layers: tp.List[torch.Tensor] = None


def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def _compute_entropy(usage: torch.Tensor) -> torch.Tensor:
    # Usage is some unnormalized distribution.
    proba = usage / usage.sum()
    p_log_p = torch.where(
        proba == 0, zero_scalar(usage.device), proba * torch.log(proba)
    )
    return -p_log_p.sum()


def _is_distributed() -> bool:
    # Checks if we need to use distributed routines.
    return distributed.is_initialized() and distributed.get_world_size() > 1


def _run_kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 50) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    # Kmeans algorithm used to initialize the codebooks.
    dim = samples.shape[-1]
    means = _sample_vectors(samples, num_clusters)
    bins = None

    for _ in range(num_iters):
        dists = torch.cdist(samples[None], means[None], p=2)[0]
        buckets = dists.argmin(dim=-1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins.clamp_(min=1)

        new_means = torch.zeros_like(means)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means /= bins[..., None]
        resampled = _sample_vectors(samples, num_clusters)
        means = torch.where(zero_mask[..., None], resampled, new_means)

    assert bins is not None
    return means, bins


def zero_scalar(device) -> torch.Tensor:
    """Returns a 0. value on the given device without introducing a synchronization point."""
    return torch.zeros([1], device=device)[0]


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.

    Buffers:
        cluster_usage (torch.Tensor): EMA of the cluster usage per batch, e.g. this will
            be dependent on the batch size etc.
        embedding_sum (torch.Tensor): EMA of the sum of the assigned points to each cluster.
            In particular, this can be normalized by `cluster_usage` to obtain the
            actual cluster centroids.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        check_unused_every: int = 5,
    ):
        super().__init__()
        self.decay = decay

        self.dim = dim
        self.codebook_size = codebook_size

        self.epsilon = epsilon
        self.threshold_usage_ratio = threshold_usage_ratio
        self.replaced_usage_ratio = replaced_usage_ratio
        self.check_unused_every = check_unused_every
        self._next_unused_check = check_unused_every
        self._cached_initialized = False

        self.register_buffer("_initialized", torch.tensor([False], dtype=torch.float))
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        embedding = torch.zeros(codebook_size, dim)
        self.register_buffer("embedding_sum", embedding)
        self.register_buffer("_embedding", None, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # Mapping old names to new names
        mappings = {
            "inited": "_initialized",
            "cluster_size": "cluster_usage",
            "embed_avg": "embedding_sum",
            "embed_sum": "embedding_sum",
        }
        for old_name, new_name in mappings.items():
            old_name = prefix + old_name
            if old_name in state_dict:
                value = state_dict.pop(old_name)
                if new_name is not None:
                    state_dict[prefix + new_name] = value
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = (
                self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            )
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    @property
    def initialized(self) -> bool:
        """Cached version of self._initialized,
        This assumes that once the module is initialized, it will never go back to the uninitialized state."""
        if not self._cached_initialized:
            self._cached_initialized = self._initialized.item()
        return self._cached_initialized

    def _init_embedding(self, data: torch.Tensor) -> None:
        # Initialize the codebook, e.g. using kmeans.
        if self.initialized:
            return

        rank = 0
        if _is_distributed():
            rank = distributed.get_rank()
            # First gathering shapes in case not all GPUs have the same effective batch size.
            # then gathering the actual content.
            if rank == 0:
                other_shapes: tp.List[torch.Size] = [None] * distributed.get_world_size()  # type: ignore
                distributed.gather_object(data.shape, other_shapes)
                other_data: tp.List[torch.Tensor] = [
                    torch.empty(shape, device=data.device, dtype=data.dtype) for shape in other_shapes]
                distributed.gather(data, other_data)
                data = torch.cat(other_data, dim=0)
            else:
                distributed.gather_object(data.shape)
                distributed.gather(data)
        if rank == 0:
            embedding, cluster_usage = _run_kmeans(data, self.codebook_size)
            self.embedding_sum.data.copy_(embedding * cluster_usage[:, None])
            self.cluster_usage.data.copy_(cluster_usage)
            self._initialized.data.fill_(1)
        # Make sure all buffers across workers are in sync after initialization
        self._broadcast_buffers()

    def _broadcast_buffers(self) -> None:
        if _is_distributed():
            for buffer in self.buffers():
                distributed.broadcast(buffer, 0)

    def _replace_expired_codes(self, samples: torch.Tensor, mask: torch.Tensor) -> None:
        # Replaces expired centroids, as indicated by `mask` (a true value indicate the code needs to be replaced).
        # The new codes are sampled from the batch `samples`.
        new_vectors = _sample_vectors(samples, self.codebook_size)
        replace_cluster_usage = (
            self.replaced_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        )
        self.embedding_sum[:] = torch.where(
            mask[:, None], replace_cluster_usage * new_vectors, self.embedding_sum
        )
        self.cluster_usage[:] = torch.where(
            mask, replace_cluster_usage, self.cluster_usage
        )

    def _check_expired_codes(self, batch_samples: torch.Tensor) -> torch.Tensor:
        # Checks whether some centroids are under utilized, and replace them if necessary.
        if not self.initialized:
            return zero_scalar(batch_samples.device)

        self._next_unused_check -= 1
        if self._next_unused_check > 0:
            return zero_scalar(batch_samples.device)
        # we don't check every iteration to avoid having too many sync points.
        self._next_unused_check = self.check_unused_every
        threshold_cluster_usage = self.threshold_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        expired_codes = self.cluster_usage < threshold_cluster_usage

        assert batch_samples.dim() == 2
        self._replace_expired_codes(batch_samples, mask=expired_codes)
        self._broadcast_buffers()

        return expired_codes.float().mean()

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        # Flattens all the dimensions but the last one, e.g. return a vector of shape `[N, D]`.
        x = rearrange(x, "... d -> (...) d")
        return x

    def _reshape_codes(self, codes: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return codes.view(*shape[:-1])

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        # Projects each vector in `x` over the nearest centroid and return its index.
        # `x` should be `[N, D]` with `N` the number of input vectors and `D` the dimension.
        assert x.dim() == 2
        dists = torch.cdist(x[None], self.embedding[None], p=2)[0]
        codes = dists.argmin(dim=-1)
        return codes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Given a tensor `x` of shape `[*, D]`, returns a tensor of integer codes of shape `[*]`.
        The codes are defined as the indexes of the centroids nearest to each vector in `x`.
        """
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        shape = x.shape
        x = self._reshape_input(x)
        codes = self._quantize(x)
        codes = self._reshape_codes(codes, shape)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Given a tensor of codes of shape `[*]`, returns a tensor of shape `[*, D]`,
        corresponding to the centroids associated to each code index.
        """
        assert (
            not codes.dtype.is_floating_point
        ), f"Codes should be integers, got {codes.dtype}"
        quantized = F.embedding(codes, self.embedding)
        return quantized

    def forward(
        self, x: torch.Tensor, initialize: bool = True
    ) -> _CodebookForwardResult:
        shape = x.shape
        x = self._reshape_input(x)

        if self.training and initialize:
            # If initialize is False, we are not allowed to initialize this layer
            # and the rest of the code will operate on a 0 filled codebook.
            # This is due to previous layers having used the batch to run kmeans init
            # and thus, the residuals are mostly 0s.
            self._init_embedding(x.detach())

        flat_codes = self._quantize(x)
        codes = self._reshape_codes(flat_codes, shape)
        quantized = self.decode(codes)
        metrics: tp.Dict[str, torch.Tensor] = {}

        if self.training:
            # We do the expiry of the unused codes at this point as buffers are in sync
            # and all the workers will take the same decision.
            expired = self._check_expired_codes(x)
            metrics['rvq_expired'] = expired
            cluster_usage = torch.zeros_like(self.cluster_usage)
            cluster_usage.scatter_add_(
                0, flat_codes, torch.ones_like(flat_codes, dtype=cluster_usage.dtype))
            _ema_inplace(self.cluster_usage, cluster_usage, self.decay)

            if self.initialized:
                # We report the entropy normalized by that of the uniform distribution,
                # This means the codebooks are optimally used when entropy=1.
                metrics['rvq_entropy'] = _compute_entropy(self.cluster_usage) / math.log(self.codebook_size)

            embedding_sum = torch.zeros_like(self.embedding_sum).to(x.dtype)
            embedding_sum.scatter_add_(0, repeat(flat_codes, "n -> n d", d=self.dim), x)
            _ema_inplace(self.embedding_sum, embedding_sum, self.decay)
            self.register_buffer('_embedding', None)

        return _CodebookForwardResult(quantized, codes, metrics)


def identity(t):
    return t

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    dim = -1,
    training = True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    π1 = (logits / temperature).softmax(dim = dim)
    one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot

def laplace_smoothing(x, n_categories, eps = 1e-5, dim = -1):
    denom = x.sum(dim = dim, keepdim = True)
    return (x + eps) / (denom + n_categories * eps)

def cdist(x, y, eps = 1e-8):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min = eps).sqrt()

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    num_probs = len(probs)

    for i, prob in enumerate(probs):
        is_last = i == (num_probs - 1)

        s = torch.binomial(total_count, prob / remainder) if not is_last else total_count
        sample[i] = s
        total_count -= s
        remainder -= prob

    assert total_count == 0, f'invalid total count {total_count}'

    return sample.to(device)

def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def append_dims_to(t, ndims):
    assert t.ndim <= ndims
    append_ndims = ndims - t.ndim
    shape = t.shape
    return t.reshape(*shape, *((1,) * append_ndims))

def accum_grad_(t, grad):
    if exists(t.grad):
        t.grad.add_(grad)
    else:
        t.grad = grad.clone().detach()
        
def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins


def ema_inplace(old, new, decay, weight = None):

    # if old.grad is populated, add it to new and set it to None

    if exists(old.grad):
        new.add_(old.grad)
        old.grad = None

    # take care of custom weighting

    weight = default(weight, 1.)

    if is_tensor(weight):
        if weight.ndim == 1:
            weight = rearrange(weight, 'c -> 1 c')

        assert weight.ndim == 2 and weight.shape == old.shape[:2]
        weight = append_dims_to(weight, old.ndim)

    old.data.lerp_(new, (1. - decay) * weight)

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

import os
import torch.distributed as dist
def measure_perplexity(predicted_indices, n_embed):
   
    # encodings = F.one_hot(predicted_indices.to(torch.int64), n_embed).float().reshape(-1, n_embed).contiguous()
    # avg_probs = encodings.mean(0)
    # perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    # cluster_use = torch.sum(avg_probs > 0)
    # return perplexity, cluster_use
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    try:
        if world_size == 1:
            encodings = F.one_hot(predicted_indices.to(torch.int64), n_embed).float().reshape(-1, n_embed).contiguous()
            avg_probs = encodings.mean(0)
            perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
            cluster_use = torch.sum(avg_probs > 0)
            return perplexity, cluster_use
        else:
            # predicted_indices:  bs * shards/卡数
            predicted_indices_list = [torch.empty_like(predicted_indices) for _ in range(world_size)]
            dist.all_gather(predicted_indices_list, predicted_indices)
            # 在主卡上生成 one-hot 编码
            if rank == 0:
                all_predicted_indices = torch.cat(predicted_indices_list, dim=0)
                #contiguous 确保张量在内存中是连续存储的
                encodings = F.one_hot(all_predicted_indices.to(torch.int64), n_embed).float().reshape(-1, n_embed).contiguous()
                avg_probs = encodings.mean(0)
                perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
                cluster_use = torch.sum(avg_probs > 0)
            else:
                perplexity = torch.tensor(0.0).cuda(local_rank)
                cluster_use = torch.tensor(0).cuda(local_rank)
            # 将 perplexity 和 cluster_use 广播回所有卡
            dist.broadcast(perplexity, src=0)
            dist.broadcast(cluster_use, src=0)
            return perplexity, cluster_use
    except Exception as e:
        print(f"An error occurred during distributed communication: {e}")
        return None, None

class EuclideanCodebook_EMA_New(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        manual_ema_update = False,
        affine_param = False,
        sync_affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update
        self.manual_ema_update = manual_ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        self.replace_sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.ones(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        # affine related params

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)

        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.update_ema()
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask = None):
        assert self.affine_param

        var_fn = partial(torch.var, unbiased = False)

        # calculate codebook mean and variance

        embed = rearrange(embed, 'h ... d -> h (...) d')

        if self.training:
            self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn), self.affine_param_codebook_decay)

        # prepare batch data, which depends on whether it has masking

        data = rearrange(data, 'h ... d -> h (...) d')

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        # calculate batch mean and variance

        if not self.sync_affine_param:
            self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
            return

        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # number of vectors, for denominator

        num_vectors = torch.tensor([num_vectors], device = device, dtype = dtype)
        distributed.all_reduce(num_vectors)

        # calculate distributed mean

        batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay('batch_mean', batch_mean, self.affine_param_batch_decay)

        # calculate distributed variance

        variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay('batch_variance', batch_variance, self.affine_param_batch_decay)

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            sampled = self.replace_sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')

            self.embed.data[ind][mask] = sampled
            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    def update_ema(self):
        cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed_normalized)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False,
        codebook_transform_fn: Callable | None = None,
        ema_update_weight: Tensor | Callable | None = None,
        accum_ema_update = False
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype
        flatten, unpack_one = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask = mask)

        # get maybe learnable codes

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        embed = embed.to(dtype)

        # affine params
        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min = 1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min = 1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        # handle maybe implicit neural codebook
        # and calculate distance

        if exists(codebook_transform_fn):
            transformed_embed = codebook_transform_fn(embed)
            transformed_embed = rearrange(transformed_embed, 'h b n c d -> h (b n) c d')
            broadcastable_input = rearrange(flatten, '... d -> ... 1 d')

            dist = -F.pairwise_distance(broadcastable_input, transformed_embed)
        else:
            dist = -cdist(flatten, embed)

        # sample or argmax depending on temperature

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)

        embed_ind = unpack_one(embed_ind, 'h *')

        if exists(codebook_transform_fn):
            transformed_embed = unpack_one(transformed_embed, 'h * c d')

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, 'h * c')

            if exists(codebook_transform_fn):
                quantize = einsum('h b n c, h b n c d -> h b n d', unpacked_onehot, transformed_embed)
            else:
                quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)

        else:
            if exists(codebook_transform_fn):
                # quantize = einx.get_at('h b n [c] d, h b n -> h b n d', transformed_embed, embed_ind)

                repeated_embed_ind = repeat(embed_ind, 'h b n -> h b n 1 d', d = transformed_embed.shape[-1])
                quantize = transformed_embed.gather(-2, repeated_embed_ind)
                quantize = rearrange(quantize, 'h b n 1 d -> h b n d')

            else:
                # quantize = einx.get_at('h [c] d, h b n -> h b n d', embed, embed_ind)

                repeated_embed = repeat(embed, 'h c d -> h b c d', b = embed_ind.shape[1])
                repeated_embed_ind = repeat(embed_ind, 'h b n -> h b n d', d = embed.shape[-1])
                quantize = repeated_embed.gather(-2, repeated_embed_ind)

        if self.training and self.ema_update and not freeze_codebook:

            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if exists(mask):
                embed_onehot[~mask] = 0.

            cluster_size = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(cluster_size)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            if callable(ema_update_weight):
                ema_update_weight = ema_update_weight(embed_sum, cluster_size)

            if accum_ema_update:
                accum_grad_(self.cluster_size, cluster_size)
                accum_grad_(self.embed_avg, embed_sum)
            else:
                ema_inplace(self.cluster_size, cluster_size, self.decay, ema_update_weight)
                ema_inplace(self.embed_avg, embed_sum, self.decay, ema_update_weight)

                if not self.manual_ema_update:
                    self.update_ema()
                    self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        dist = unpack_one(dist, 'h * d')
        metrics: tp.Dict[str, torch.Tensor] = {}
        perplexity, cluster_use = measure_perplexity(embed_ind, self.codebook_size)
        metrics['perplexity'] = perplexity
        metrics['cluster_use'] = cluster_use
        return _CodebookForwardResult(quantize, embed_ind, metrics)



class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.
    """

    def __init__(
        self,
        ema_new: bool,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        if ema_new:
            self._codebook = EuclideanCodebook_EMA_New(
                dim=codebook_dim,
                codebook_size=codebook_size,
                decay=decay,
                eps=epsilon,
            )
        else:
            self._codebook = EuclideanCodebook(
                dim=codebook_dim,
                codebook_size=codebook_size,
                decay=decay,
                epsilon=epsilon,
                threshold_usage_ratio=threshold_usage_ratio,
                **kwargs,
            )
        self.ema_new = ema_new
        self.codebook_size = codebook_size

    @property
    def embedding(self):
        return self._codebook.embedding

    @property
    def initialized(self):
        return self._codebook.initialized

    def _rearrange_input(self, x):
        x = rearrange(x, "b d n -> b n d")
        return x

    def _rearrange_output(self, quantized):
        quantized = rearrange(quantized, "b n d -> b d n")
        return quantized

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes `x` into discrete integer codes."""
        x = self._rearrange_input(x)
        x = self.project_in(x)
        codes = self._codebook.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts integer codes into quantized vectors."""
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)
        return quantized

    def forward(self, x: torch.Tensor, initialize: bool = True) -> _VQForwardResult:
        x = self._rearrange_input(x)
        
        x = self.project_in(x)
        
        quantized, codes, metrics = self._codebook(x)

        if self.training:
            quantized = x + (quantized - x).detach()
            loss = F.mse_loss(x, quantized.detach())
        else:
            loss = zero_scalar(x.device)

        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)

        return _VQForwardResult(quantized, codes, loss, metrics)


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers: int, codebook_offset: int, ema_new: bool, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(ema_new, **kwargs) for _ in range(num_quantizers)]
        )
        self.codebook_offset = codebook_offset
        self.ema_new = ema_new

    def forward(
        self, x: torch.Tensor, n_q: tp.Optional[int] = None, layers: tp.List[int] = []
    ) -> _VQForwardResult:
        """
        Args:
            x (torch.Tensor): input tensor to quantize, of shape `[B, C, T]`.
            n_q (int or None): if provided, number of codebook levels to use in RVQ.
        """

        quantized_out = zero_scalar(x.device)
        residual = x

        all_losses = []
        all_codes = []
        all_metrics: tp.Dict[str, torch.Tensor] = {}

        n_q = n_q or len(self.layers)
        previous_layer_is_initialized = True
        q_layers = []
        for i, layer in enumerate(self.layers[:n_q]):  # type: ignore
            if self.training and not self.ema_new:
                this_layer_is_initialized = layer.initialized
                
                
            # We only allow the kmeans initialization if the previous layer is already initialized from the previous
            # iterations, this is to avoid learning the subsequent kmeans on the same batch, which would eventually
            # lead to its exhaustion and running kmeans on 0 values.
            quantized, codes, loss, metrics, _ = layer(
                residual, initialize=previous_layer_is_initialized
            )
            if i in layers:
                q_layers.append(quantized)
            if self.training and not self.ema_new:
                previous_layer_is_initialized = this_layer_is_initialized  # type: ignore

            quantized = quantized.detach()
            residual = residual - quantized#.detach()
            quantized_out = quantized_out + quantized

            all_codes.append(codes)
            all_losses.append(loss)

            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key] += value / n_q
                else:
                    all_metrics[key] = value / n_q
                all_metrics[key + f"_{i + self.codebook_offset}"] = value

        if self.training:
            # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
            quantized_out = x + (quantized_out - x).detach()

        out_losses, out_codes = map(torch.stack, (all_losses, all_codes))
        return _VQForwardResult(quantized_out, out_codes, out_losses, all_metrics, q_layers)

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        """Encodes `x` into discrete integer codes. If `n_q` is provided, only uses the first `n_q` codebook levels."""
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:  # type: ignore
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts the integer codes into quantized vectors."""
        quantized = zero_scalar(codes.device)
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            quantized = quantized + layer.decode(layer_codes)
        return quantized
