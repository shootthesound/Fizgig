"""Every captioner Fizgig offers, in the model menu's order. A new captioner is a description here plus its own file in
this folder (docs/ADDING_A_CAPTIONER.md); nothing else in Fizgig changes for it."""
from fizgig.captioners import CaptionerDescription, HFRepo, PrefFile

_FLORENCE_TASKS = ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>")
_FLORENCE = "fizgig.captioners.florence:FlorenceCaptioner"

CAPTIONERS = (
    CaptionerDescription(
        key="florence2_promptgen", label="MiaoshouAI/Florence-2-base-PromptGen", kind="sentences", module=_FLORENCE,
        files=HFRepo("MiaoshouAI/Florence-2-base-PromptGen", "da7ac9f3deac56a928e2fd4d94d8bb985d231299",
                     code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"),
        licence="MIT", vram_gb=2.5, size_gb=0.5, tasks=_FLORENCE_TASKS, default_task="<DETAILED_CAPTION>"),
    CaptionerDescription(
        key="florence2_base", label="microsoft/Florence-2-base", kind="sentences", module=_FLORENCE,
        files=HFRepo("microsoft/Florence-2-base", "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"),
        licence="MIT", vram_gb=2.5, size_gb=0.5, tasks=_FLORENCE_TASKS, default_task="<DETAILED_CAPTION>"),
    CaptionerDescription(
        key="florence2_large", label="microsoft/Florence-2-large", kind="sentences", module=_FLORENCE,
        files=HFRepo("microsoft/Florence-2-large", "21a599d414c4d928c9032694c424fb94458e3594"),
        licence="MIT", vram_gb=2.5, size_gb=1.5, tasks=_FLORENCE_TASKS, default_task="<DETAILED_CAPTION>"),
    CaptionerDescription(
        key="qwen3vl", label="Qwen3-VL (Krea 2 text encoder, or the secondary captioning model)", kind="sentences",
        module="fizgig.captioners.qwen3vl:Qwen3VLCaptioner", files=PrefFile("captioner"),
        licence="Apache-2.0", vram_gb=8.0, instruction=True),
    CaptionerDescription(
        key="wd_eva02_large_v3", label="WD EVA02 Large v3 (booru tags)", kind="tags",
        module="fizgig.captioners.wd_tagger:WDTagger",
        files=HFRepo("SmilingWolf/wd-eva02-large-tagger-v3", "b25b82a03f7282e41aa2f257a52c7583b710bd1c"),
        licence="Apache-2.0", vram_gb=2.0, size_gb=1.3,
        hint="Booru tags (1girl, long hair, smile...) for anime and illustration: what Anima and anime SDXL "
             "models were trained on. Downloads itself the first time (1.3 GB)."),
)
