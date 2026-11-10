"""Florence-2 (Microsoft's, and MiaoshouAI's PromptGen fine-tune): a short fixed task menu, no instruction."""
import torch

from fizgig.captioners import Captioner, CaptionRequest, CaptionResult


class FlorenceCaptioner(Captioner):
    @classmethod
    def fetch(cls, desc, files):
        """Its files, then its processor and config the way load() asks for them: Florence-2 runs its own code
        (trust_remote_code), which transformers caches on first use - without that cache, offline, transformers
        falls back to its built-in Florence-2, which this model's processor doesn't match."""
        super().fetch(desc, files)
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        kw = {"revision": files["revision"]}
        if files.get("code_revision"):
            kw["code_revision"] = files["code_revision"]
        cfg = AutoConfig.from_pretrained(files["repo"], trust_remote_code=True, **kw)
        AutoProcessor.from_pretrained(files["repo"], trust_remote_code=True, **kw)
        for ref in (getattr(cfg, "auto_map", None) or {}).values():     # the model class's code (maybe another repo)
            get_class_from_dynamic_module(ref, files["repo"], **kw)

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        from fizgig.utils.hf_cache import from_pretrained_cache_first
        kw = {"trust_remote_code": True, "revision": self.files["revision"]}
        if self.files.get("code_revision"):
            kw["code_revision"] = self.files["code_revision"]
        self.processor = from_pretrained_cache_first(AutoProcessor, self.files["repo"], **kw)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # torch_dtype, not dtype: the pinned transformers reads torch_dtype; an unknown kwarg falls through silently
        # and Florence loads fp32 (double VRAM), which nothing reports.
        self.model = from_pretrained_cache_first(AutoModelForCausalLM, self.files["repo"], torch_dtype=dtype,
                                                 attn_implementation="eager", **kw).to(self.device).eval()

    def caption(self, image, request: CaptionRequest) -> CaptionResult:
        task = request.task if request.task in self.desc.tasks else self.desc.default_task
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                  max_new_tokens=request.max_new_tokens, do_sample=False, num_beams=3, use_cache=False)
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(raw, task=task, image_size=(image.width, image.height))
        return CaptionResult(text=parsed.get(task, raw))

    def unload(self):
        self.model = self.processor = None
