# HuggingFaceTB/SmolLM3-3B-Base and the Open-MOPD RL teacher family.
# Megatron's no_rope_freq=4 skips RoPE on layers 4,8,...,36;
# HF's no_rope_layers uses the inverse convention [1,1,1,0]*9.
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2048
   --ffn-hidden-size 11008
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 4
   --max-position-embeddings 65536
   --use-rotary-position-embeddings
   --no-rope-freq 4
   --disable-bias-linear
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --rotary-base 5000000
   --vocab-size 128256
   --kv-channels 128
)
