pub const ALLOW: &[&str] = &[
  "blocks.*.attn.q_proj", "blocks.*.attn.k_proj", "blocks.*.attn.v_proj", "blocks.*.attn.o_proj",
  "blocks.*.mlp.fc1", "blocks.*.mlp.fc2",
  "resblocks.*.conv*",
];
pub const DENY: &[&str] = &[
  "distilled_guidance_layer.*",
];

