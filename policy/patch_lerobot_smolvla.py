# patch_lerobot_smolvla.py: S.H.E.A.R's edits to the INSTALLED lerobot's SmolVLA (K7 2026-11-06). The Colab/Kaggle
# notebook (policy/colab_train_v3.ipynb, cell 3) embeds this same code, from the next line down; on the laptop:
# python policy/patch_lerobot_smolvla.py in the lerobot env. It rewrites the installed lerobot file in place.
# Pinned lerobot commit: 5aa74557 (the anchors below are checked against it).
# The SHEAR edits to lerobot's SmolVLA (the same as the laptop's local lerobot, K7 2026-11-06).
#    a) Upstream, train_expert_only freezes the WHOLE vision-language model, vision encoder included, so
#       --policy.freeze_vision_encoder=false did nothing. Now the vision encoder trains with the action expert and
#       the language layers stay frozen (186M of 450M params train).
#    b) SmolVLA loads the vision-language model and the expert in bfloat16. The trainable weights are kept in
#       float32 instead (fp16 still does the fast compute on the T4): the fp16 gradient scaler can't handle
#       bfloat16 gradients ("not implemented for 'BFloat16'"), and small updates get lost in bfloat16 weights.
#    Safe to run again: each edit checks whether it is already there.
import pathlib

import lerobot.policies.smolvla.smolvlm_with_expert as m

PATH = pathlib.Path(m.__file__)
s = PATH.read_text()

def put(s, anchor, new, where):
    n = s.count(anchor)
    assert n == 1, f"anchor found {n} times, expected once: the lerobot version changed"
    return s.replace(anchor, new + anchor if where == "before" else anchor + new)

edits = []
if "if self.train_expert_only and not self.freeze_vision_encoder:" not in s:
    s = put(s, "        # To avoid unused params issue with distributed training\n"
               "        for name, params in self.lm_expert.named_parameters():\n",
            "        # SHEAR: the vision encoder trains with the expert\n"
            "        if self.train_expert_only and not self.freeze_vision_encoder:\n"
            "            self.get_vlm_model().vision_model.train()\n"
            "            for params in self.get_vlm_model().vision_model.parameters():\n"
            "                params.requires_grad = True\n", "before")
    edits.append("vision encoder trainable")
if "vision_model.train(mode)" not in s:
    s = put(s, "            self.get_vlm_model().vision_model.eval()\n\n"
               "        if self.train_expert_only:\n            self.vlm.eval()\n",
            "            if not self.freeze_vision_encoder:\n"
            "                self.get_vlm_model().vision_model.train(mode)\n", "after")
    edits.append("vision encoder in train mode")
if "SHEAR fp32 master weights" not in s:
    s = put(s, "        for name, params in self.lm_expert.named_parameters():\n"
               "            if \"lm_head\" in name:\n"
               "                params.requires_grad = False\n",
            "        # SHEAR fp32 master weights: trainable weights in float32, the frozen ones stay bfloat16\n"
            "        for params in self.parameters():\n"
            "            if params.requires_grad and params.dtype == torch.bfloat16:\n"
            "                params.data = params.data.float()\n", "after")
    edits.append("trainable weights in float32")
if "SHEAR connector dtype" not in s:
    s = put(s, "        # Modality projection & resampling\n",
            "        # SHEAR connector dtype: the float32 vision encoder feeds the frozen bfloat16 connector\n"
            "        image_hidden_states = image_hidden_states.to(\n"
            "            dtype=next(self.get_vlm_model().connector.parameters()).dtype\n"
            "        )\n", "before")
    edits.append("connector input cast")
compile(s, str(PATH), "exec")
PATH.write_text(s)
print("patched:" if edits else "already patched:", ", ".join(edits) or "all four edits present", "->", PATH)
