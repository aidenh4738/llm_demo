"""
Decoder-only 架构推理流程演示（含 KV Cache）
=========================================

用最小的、可运行的 numpy 实现，完整演示一次大模型推理的两个阶段：
  1) Prefill（预填充）：一次性处理完整 prompt，为每一层建立 KV 缓存
  2) Decode（解码）：每一步只处理新生成的 1 个 token，复用 KV 缓存，逐个生成

注意：推理只有前向计算，没有反向传播；这里的权重是随机初始化的，
只是为了演示"数据怎么流动、缓存怎么变化"，不代表真实语言能力。
"""

import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 模型超参数（故意调得很小，便于把每一步的张量形状看清楚）
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50      # 词表大小
D_MODEL = 32         # 隐藏维度
N_HEADS = 4          # 注意力头数
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 2         # decoder block 层数
D_FF = 64            # FFN中间维度
MAX_LEN = 64         # 支持的最大序列长度（用于位置编码表大小）


def init_params():
    """随机初始化所有权重，结构对应：embedding -> N x decoder block -> 输出头"""
    p = {
        "tok_emb": np.random.randn(VOCAB_SIZE, D_MODEL) * 0.02,
        "pos_emb": np.random.randn(MAX_LEN, D_MODEL) * 0.02,
        "layers": [],
        "ln_f_g": np.ones(D_MODEL),
        "ln_f_b": np.zeros(D_MODEL),
        "head": np.random.randn(D_MODEL, VOCAB_SIZE) * 0.02,
    }
    for _ in range(N_LAYERS):
        p["layers"].append({
            "ln1_g": np.ones(D_MODEL), "ln1_b": np.zeros(D_MODEL),
            "Wq": np.random.randn(D_MODEL, D_MODEL) * 0.02,
            "Wk": np.random.randn(D_MODEL, D_MODEL) * 0.02,
            "Wv": np.random.randn(D_MODEL, D_MODEL) * 0.02,
            "Wo": np.random.randn(D_MODEL, D_MODEL) * 0.02,
            "ln2_g": np.ones(D_MODEL), "ln2_b": np.zeros(D_MODEL),
            "W1": np.random.randn(D_MODEL, D_FF) * 0.02, "b1": np.zeros(D_FF),
            "W2": np.random.randn(D_FF, D_MODEL) * 0.02, "b2": np.zeros(D_MODEL),
        })
    return p


# ---------------------------------------------------------------------------
# KV Cache：每一层各自维护一份，形状 [n_heads, 已缓存的seq_len, head_dim]
# ---------------------------------------------------------------------------

class KVCache:
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def append(self, layer_idx, new_k, new_v):
        """把这一次新算出来的K,V拼接到该层已有的缓存后面"""
        if self.k[layer_idx] is None:
            self.k[layer_idx] = new_k
            self.v[layer_idx] = new_v
        else:
            self.k[layer_idx] = np.concatenate([self.k[layer_idx], new_k], axis=1)
            self.v[layer_idx] = np.concatenate([self.v[layer_idx], new_v], axis=1)
        return self.k[layer_idx], self.v[layer_idx]

    def cached_len(self, layer_idx):
        return 0 if self.k[layer_idx] is None else self.k[layer_idx].shape[1]


def split_heads(x):
    seq = x.shape[0]
    return x.reshape(seq, N_HEADS, HEAD_DIM).transpose(1, 0, 2)  # [n_heads, seq, head_dim]


def merge_heads(x):
    x = x.transpose(1, 0, 2)
    return x.reshape(x.shape[0], D_MODEL)


# ---------------------------------------------------------------------------
# 因果自注意力（核心：这里读/写KV缓存）
# ---------------------------------------------------------------------------

def causal_self_attention(x, layer, cache, layer_idx):
    """
    x: [new_len, D_MODEL] —— 这一次前向"新"要处理的token
       prefill时 new_len = prompt长度；decode时 new_len = 1
    """
    new_len = x.shape[0]
    q = split_heads(x @ layer["Wq"])
    k_new = split_heads(x @ layer["Wk"])
    v_new = split_heads(x @ layer["Wv"])

    past_len = cache.cached_len(layer_idx)
    k_full, v_full = cache.append(layer_idx, k_new, v_new)   # 写缓存
    total_len = past_len + new_len

    # attention分数: [n_heads, new_len, total_len]
    scores = np.einsum("hqd,hkd->hqk", q, k_full) / np.sqrt(HEAD_DIM)

    # 因果mask：新token里第i个（绝对位置 past_len+i）只能看到 <= 自己位置的key
    mask = np.full((new_len, total_len), -np.inf)
    for i in range(new_len):
        abs_pos = past_len + i
        mask[i, : abs_pos + 1] = 0.0
    scores = scores + mask

    attn = softmax(scores, axis=-1)
    out = np.einsum("hqk,hkd->hqd", attn, v_full)             # 读缓存后的attention输出
    return merge_heads(out) @ layer["Wo"]


def decoder_forward(token_ids, params, cache, start_pos):
    """跑完整个decoder-only网络一次，返回这批token各自的logits"""
    x = params["tok_emb"][token_ids] + params["pos_emb"][start_pos: start_pos + len(token_ids)]

    for i, layer in enumerate(params["layers"]):
        attn_in = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
        x = x + causal_self_attention(attn_in, layer, cache, i)        # 残差1

        ff_in = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
        h = gelu(ff_in @ layer["W1"] + layer["b1"])
        x = x + (h @ layer["W2"] + layer["b2"])                        # 残差2

    x = layer_norm(x, params["ln_f_g"], params["ln_f_b"])
    return x @ params["head"]   # logits: [new_len, VOCAB_SIZE]


def pick_next(logits_last):
    """贪婪解码：直接取概率最大的token。
    用贪婪而不是随机采样，是为了让"有缓存"和"无缓存"两条路径的结果可以严格对比——
    因为前向计算是确定性的，缓存只影响计算量，不应该改变最终结果。"""
    return int(np.argmax(logits_last))


# ---------------------------------------------------------------------------
# 完整推理流程：Prefill + Decode（带KV缓存）
# ---------------------------------------------------------------------------

def generate_with_cache(prompt_ids, n_new_tokens, params):
    cache = KVCache(N_LAYERS)
    generated = list(prompt_ids)
    token_forward_count = 0  # 统计每一步真正"新算"的token数，用于和不用缓存对比

    print(f"\n[Prefill] 一次性处理完整prompt，共 {len(prompt_ids)} 个token：{prompt_ids}")
    logits = decoder_forward(prompt_ids, params, cache, start_pos=0)
    token_forward_count += len(prompt_ids)
    print(f"  -> 每层KV缓存形状: {cache.k[0].shape}  (n_heads={N_HEADS}, seq_len={cache.cached_len(0)}, head_dim={HEAD_DIM})")

    next_id = pick_next(logits[-1])
    generated.append(next_id)
    print(f"  -> 取最后一个位置的logits做贪婪解码，得到第1个新token: {next_id}")

    print(f"\n[Decode] 逐个生成剩下 {n_new_tokens - 1} 个token，每步只新算1个token：")
    for step in range(n_new_tokens - 1):
        cur_pos = len(generated) - 1  # 这个新token在整段序列里的绝对位置
        logits = decoder_forward([generated[-1]], params, cache, start_pos=cur_pos)
        token_forward_count += 1
        next_id = pick_next(logits[-1])
        generated.append(next_id)
        print(f"  step {step+1}: 输入1个token(绝对位置{cur_pos}) -> 缓存长度变为{cache.cached_len(0)} -> 新token: {next_id}")

    return generated, token_forward_count


def generate_without_cache(prompt_ids, n_new_tokens, params):
    """对照组：完全不用KV缓存，每一步都把"已生成的全部序列"重新算一遍"""
    generated = list(prompt_ids)
    token_forward_count = 0

    for step in range(n_new_tokens):
        fresh_cache = KVCache(N_LAYERS)  # 每步都是全新的、空的缓存 = 等价于不缓存
        logits = decoder_forward(generated, params, fresh_cache, start_pos=0)
        token_forward_count += len(generated)  # 这一步重新算了整段序列
        next_id = pick_next(logits[-1])
        generated.append(next_id)

    return generated, token_forward_count


if __name__ == "__main__":
    params = init_params()

    prompt = [3, 17, 8, 42, 5]   # 假装是5个token的prompt（这里直接用随机id代表）
    N_NEW = 6                    # 要新生成6个token

    print("=" * 70)
    print("用KV缓存的完整推理流程")
    print("=" * 70)
    result_cached, cost_cached = generate_with_cache(prompt, N_NEW, params)
    print(f"\n最终生成序列: {result_cached}")

    print("\n" + "=" * 70)
    print("不用KV缓存的对照流程（每步重新计算整段序列）")
    print("=" * 70)
    result_nocache, cost_nocache = generate_without_cache(prompt, N_NEW, params)
    print(f"最终生成序列: {result_nocache}")

    print("\n" + "=" * 70)
    print("一致性校验 + 计算量对比")
    print("=" * 70)
    same = (result_cached == result_nocache)
    print(f"  两条路径生成结果是否完全一致: {same}  （应为True：缓存只优化速度，不改变前向计算结果）")
    print(f"  用KV缓存:   累计处理 {cost_cached} 个 token-前向     (= prefill长度 + 后续每步1个)")
    print(f"  不用缓存:   累计处理 {cost_nocache} 个 token-前向    (= 每步都重算已有的全部序列)")
    print(f"  节省倍数:   {cost_nocache / cost_cached:.2f}x")
