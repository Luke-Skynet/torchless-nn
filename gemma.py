from utils import init_random_tensor
from layers import RMSNorm, RotaryEmbedding, Softcap, TransformerBlock, gemma_layer_types
from activations import GeLU, SoftMax
from transformer_adapters import GPTEmbedFront, GPTEmbedBack
from network import Network


# The published Gemma 4 31B text configuration, for reference and for scaling down.
# Sanity check: gemma_parameter_count(**GEMMA_4_31B) lands on the ~30.7B text params
# implied by the checkpoint's 62,546,177,752 bf16 bytes once the vision tower is set aside.
GEMMA_4_31B = dict(
    vocab_size            = 262144,
    context_length        = 262144,
    num_layers            = 60,
    embed_dim             = 5376,
    ffn_multiplier        = 4,        # 21504 intermediate
    num_heads             = 32,
    head_dim              = 256,
    num_kv_heads          = 16,
    global_head_dim       = 512,      # global layers widen keys and queries
    num_global_kv_heads   = 4,
    sliding_window        = 1024,
    pattern               = 6,        # 5 sliding : 1 global
    local_theta           = 10000.0,
    global_theta          = 1000000.0,
    global_partial_rotary = 0.25,     # p-RoPE, global layers only
    logit_softcap         = 30.0,
    qk_norm               = True,
    post_norm             = True,
    kv_shared_global      = True,     # attention_k_eq_v
)


def gemma_gpt(vocab_size, context_length, num_layers, embed_dim, num_heads, head_dim,
              num_kv_heads, global_head_dim = None, num_global_kv_heads = None,
              sliding_window = 1024, pattern = 6, ffn_multiplier = 4,
              local_theta = 10000.0, global_theta = 1000000.0, global_partial_rotary = 0.25,
              logit_softcap = 30.0, qk_norm = True, post_norm = True, kv_shared_global = True,
              chunk_size = None, dropout_rate = 0.0, activation = GeLU, embedding_table = None):

    """Assemble a Gemma 4 shaped decoder.

    Local layers use sliding window attention with full rotation RoPE. Global layers
    attend over the whole sequence, widen their heads, share one projection between
    keys and values, and use p-RoPE so their low frequency dimensions carry no
    position signal. gemma_layer_types decides which is which.

    Both RoPE tables are built once and shared across every layer that uses them:
    they hold no parameters, only cached cos/sin.

    chunk_size sets the query block size for attention, defaulting to the sliding
    window. That is what turns the window into an actual saving rather than just a
    mask: a sliding layer then only computes scores against the keys in range,
    instead of computing the full square and discarding most of it. Pass
    chunk_size = 0 to disable chunking and compute each sequence in one block.
    """

    if chunk_size is None:
        chunk_size = sliding_window

    global_head_dim     = head_dim     if global_head_dim     is None else global_head_dim
    num_global_kv_heads = num_kv_heads if num_global_kv_heads is None else num_global_kv_heads

    if embedding_table is None:
        # scaled so that the sqrt(embed_dim) multiply in GPTEmbedFront lands the
        # embeddings at unit scale on the residual stream
        embedding_table = init_random_tensor((vocab_size, embed_dim)) / embed_dim**0.5

    local_rope  = RotaryEmbedding(head_dim, context_length, theta = local_theta)
    global_rope = RotaryEmbedding(global_head_dim, context_length, theta = global_theta,
                                  partial_rotary_factor = global_partial_rotary)

    stack = [GPTEmbedFront(embedding_table, context_length,
                           positional = "none", scale_embeddings = True)]

    for kind in gemma_layer_types(num_layers, pattern):
        is_global = kind == "global"
        stack.append(TransformerBlock(
            embed_dim, context_length, num_heads, activation, decoder = True,
            dropout_rate = dropout_rate, norm = RMSNorm, post_norm = post_norm,
            ffn_multiplier = ffn_multiplier,
            num_kv_heads   = num_global_kv_heads if is_global else num_kv_heads,
            head_dim       = global_head_dim     if is_global else head_dim,
            qk_norm        = qk_norm,
            rope           = global_rope         if is_global else local_rope,
            sliding_window = None                if is_global else sliding_window,
            kv_shared      = kv_shared_global and is_global,
            chunk_size     = chunk_size or None))

    stack.append(RMSNorm(embed_dim))
    stack.append(GPTEmbedBack(embedding_table))

    if logit_softcap:
        stack.append(Softcap(logit_softcap))

    stack.append(SoftMax())

    return Network(stack)


def gemma_parameter_count(vocab_size, num_layers, embed_dim, num_heads, head_dim, num_kv_heads,
                          global_head_dim = None, num_global_kv_heads = None, pattern = 6,
                          ffn_multiplier = 4, qk_norm = True, post_norm = True,
                          kv_shared_global = True, **ignored):

    """Parameter count for a gemma_gpt config, without allocating anything."""

    global_head_dim     = head_dim     if global_head_dim     is None else global_head_dim
    num_global_kv_heads = num_kv_heads if num_global_kv_heads is None else num_global_kv_heads

    total = vocab_size * embed_dim + embed_dim # tied embeddings, final norm

    for kind in gemma_layer_types(num_layers, pattern):
        is_global = kind == "global"

        dim   = global_head_dim     if is_global else head_dim
        heads = num_global_kv_heads if is_global else num_kv_heads

        total += embed_dim * num_heads * dim * 2          # query and output
        total += embed_dim * heads * dim                  # key
        if not (kv_shared_global and is_global):
            total += embed_dim * heads * dim              # separate value
        if qk_norm:
            total += 2 * dim

        total += 3 * embed_dim * ffn_multiplier * embed_dim
        total += (4 if post_norm else 2) * embed_dim

    return total
