"""JoyCaption Beta One (fancyfeast): an 8B LLaVA captioner (Llama 3.1 + SigLIP 2) built for training captions, good
at people (gender, reflections, mirrors) and at every style. Follows the Captions tab's instruction, as Qwen3-VL does.

bf16 when the card has room, else 4-bit (bitsandbytes NF4) on the language model with the vision tower and projector
kept in bf16. The prompt is built exactly as the model card does it: HF's LLaVA chat handling easily doubles the <bos>
token, which makes the model worse, so the template + processor pair here is the one its author checked."""
import torch

from fizgig.captioners import Captioner, CaptionRequest, CaptionResult

SYSTEM = "You are a helpful image captioner."


class JoyCaption(Captioner):
    def load(self):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        from fizgig.utils.hf_cache import from_pretrained_cache_first
        kw = {"revision": self.files["revision"]}
        self.processor = from_pretrained_cache_first(AutoProcessor, self.files["repo"], **kw)
        if self.device != "cuda":
            mk = {"torch_dtype": torch.float32}
        elif self.precision == "nf4":
            from transformers import BitsAndBytesConfig
            mk = {"torch_dtype": torch.bfloat16, "device_map": {"": 0},
                  "quantization_config": BitsAndBytesConfig(
                      load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                      llm_int8_skip_modules=["vision_tower", "multi_modal_projector"])}
        else:
            mk = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
        self.model = from_pretrained_cache_first(LlavaForConditionalGeneration, self.files["repo"], **kw, **mk).eval()

    @torch.no_grad()
    def caption(self, image, request: CaptionRequest) -> CaptionResult:
        convo = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": request.instruction or "Write a descriptive caption for this image."}]
        text = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype if self.device != "cuda" else torch.bfloat16)
        ids = self.model.generate(**inputs, max_new_tokens=request.max_new_tokens, do_sample=True, suppress_tokens=None,
                                  use_cache=True, temperature=0.6, top_k=None, top_p=0.9)[0]
        ids = ids[inputs["input_ids"].shape[1]:]
        return CaptionResult(text=self.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False).strip())

    def unload(self):
        self.model = self.processor = None
