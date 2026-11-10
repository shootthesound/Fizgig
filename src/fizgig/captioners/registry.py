"""Every captioner Fizgig offers, in the model menu's order. A new captioner is a description here plus its own file in
this folder (docs/ADDING_A_CAPTIONER.md); nothing else in Fizgig changes for it."""
from fizgig.captioners import CaptionerDescription, HFRepo, PrefFile

_FLORENCE_TASKS = ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>")

# JoyCaption Beta One's own caption types and extra options, worded as in its author's demo (fpgaminer/joycaption,
# Apache-2.0; the Space fancyfeast/joy-caption-beta-one). Left out: its tag modes (its README calls them less stable;
# the WD tagger does tags), the product-listing and social-media types (not training captions), and the profanity
# options. Anyone can still type any instruction (Custom...).
_JC_STRAIGHT = ("Begin with the main subject and medium. Mention pivotal elements\u2014people, objects, scenery\u2014using "
                "confident, definite language. Focus on concrete details like color, shape, texture, and spatial "
                "relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote "
                "it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, "
                "resolution, or unobservable details. Vary your sentence structure and keep the description concise, "
                "without starting with \u201cThis image is\u2026\u201d or similar phrasing.")
_JC_TYPES = (
    ("Descriptive", "Write a detailed description for this image.",
     "Write a detailed description for this image in {word_count} words or less.",
     "Write a {length} detailed description for this image."),
    ("Descriptive (casual)", "Write a descriptive caption for this image in a casual tone.",
     "Write a descriptive caption for this image in a casual tone within {word_count} words.",
     "Write a {length} descriptive caption for this image in a casual tone."),
    ("Straightforward", "Write a straightforward caption for this image. " + _JC_STRAIGHT,
     "Write a straightforward caption for this image within {word_count} words. " + _JC_STRAIGHT,
     "Write a {length} straightforward caption for this image. " + _JC_STRAIGHT),
    ("Stable Diffusion prompt",
     "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
     "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. "
     "{word_count} words or less.",
     "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt."),
    ("MidJourney prompt", "Write a MidJourney prompt for this image.",
     "Write a MidJourney prompt for this image within {word_count} words.",
     "Write a {length} MidJourney prompt for this image."),
    ("Art critic",
     "Analyze this image like an art critic would with information about its composition, style, symbolism, the use "
     "of color, light, any artistic movement it might belong to, etc.",
     "Analyze this image like an art critic would with information about its composition, style, symbolism, the use "
     "of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
     "Analyze this image like an art critic would with information about its composition, style, symbolism, the use "
     "of color, light, any artistic movement it might belong to, etc. Keep it {length}."),
)
_JC_LENGTHS = ("any", "very short", "short", "medium-length", "long", "very long") + tuple(str(n) for n in range(20, 261, 20))
_JC_EXTRAS = (
    ("Call the person by the trigger word", "If there is a person/character in the image you must refer to them as "
     "{name}.", False),        # off: JoyCaption then repeats the trigger word in most sentences
    ("Lighting", "Include information about lighting.", True),
    ("Camera angle", "Include information about camera angle.", True),
    ("Shot type (close-up to wide)", "Mention whether the image depicts an extreme close-up, close-up, medium close-up, "
     "medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.", True),
    ("Not the resolution", "Do NOT mention the image's resolution.", True),
    ("No \u201cThis image shows\u2026\u201d", "Your response will be used by a text-to-image model, so avoid useless meta "
     "phrases like \u201cThis image shows\u2026\u201d, \"You are looking at...\", etc.", True),
    ("Vantage height", "Explicitly specify the vantage height (eye-level, low-angle worm\u2019s-eye, bird\u2019s-eye, "
     "drone, rooftop, etc.).", False),
    ("Depth of field", "Specify the depth of field and whether the background is in focus or blurred.", False),
    ("Light sources", "If applicable, mention the likely use of artificial or natural lighting sources.", False),
    ("Composition", "Include information on the image's composition style, such as leading lines, rule of thirds, or "
     "symmetry.", False),
    ("Orientation", "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.", False),
    ("Camera settings (photos)", "If it is a photo you MUST include information about what camera was likely used and "
     "details such as aperture, shutter speed, ISO, etc.", False),
    ("Ages", "Include information about the ages of any people/characters when applicable.", False),
    ("Not fixed traits", "Do NOT include information about people/characters that cannot be changed (like ethnicity, "
     "gender, etc), but do still include changeable attributes (like hair style).", False),
    ("Watermark: say if there is one", "Include information about whether there is a watermark or not.", False),
    ("Watermark: mention it", "If there is a watermark, you must mention it.", False),
    ("JPEG artifacts", "Include information about whether there are JPEG artifacts or not.", False),
    ("Aesthetic quality", "You MUST include information about the subjective aesthetic quality of the image from low to "
     "very high.", False),
    ("Not the text in it", "Do NOT mention any text that is in the image.", False),
    ("Not the mood", "Do not mention the mood/feeling/etc of the image.", False),
    ("No ambiguous language", "Do NOT use any ambiguous language.", False),
    ("Only the main elements", "ONLY describe the most important elements of the image.", False),
    ("No artist or title", "If it is a work of art, do not include the artist's name or the title of the work.", False),
    ("SFW / suggestive / NSFW", "Include whether the image is sfw, suggestive, or nsfw.", False),
    ("Keep it PG", "Do NOT include anything sexual; keep it PG.", False),
)
_FLORENCE = "fizgig.captioners.florence:FlorenceCaptioner"

CAPTIONERS = (
    CaptionerDescription(
        key="florence2_promptgen", label="MiaoshouAI/Florence-2-base-PromptGen", kind="sentences", module=_FLORENCE,
        files=HFRepo("MiaoshouAI/Florence-2-base-PromptGen", "da7ac9f3deac56a928e2fd4d94d8bb985d231299",
                     code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"),
        licence="MIT", vram_gb=2.5, size_gb=1.0, tasks=_FLORENCE_TASKS, default_task="<DETAILED_CAPTION>",
        blurb="Quick captions in the style of image prompts"),
    CaptionerDescription(
        key="florence2_base", label="microsoft/Florence-2-base", kind="sentences", module=_FLORENCE,
        files=HFRepo("microsoft/Florence-2-base", "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"),
        licence="MIT", vram_gb=2.5, size_gb=0.5, tasks=_FLORENCE_TASKS, default_task="<DETAILED_CAPTION>",
        blurb="Microsoft's small, quick captioner", download_skip=("pytorch_model.bin", "*.ipynb")),
    CaptionerDescription(
        key="florence2_large", label="microsoft/Florence-2-large", kind="sentences", module=_FLORENCE,
        files=HFRepo("microsoft/Florence-2-large", "21a599d414c4d928c9032694c424fb94458e3594"),
        licence="MIT", vram_gb=2.5, size_gb=1.5, tasks=_FLORENCE_TASKS, default_task="<DETAILED_CAPTION>",
        blurb="Microsoft's larger captioner: more detail", download_skip=("pytorch_model.bin", "*.ipynb")),
    CaptionerDescription(
        key="qwen3vl", label="Qwen3-VL (Krea 2 text encoder, or the secondary captioning model)", kind="sentences",
        module="fizgig.captioners.qwen3vl:Qwen3VLCaptioner", files=PrefFile("captioner"),
        licence="Apache-2.0", vram_gb=8.0, instruction=True),
    CaptionerDescription(
        key="joycaption_beta_one", label="JoyCaption Beta One", kind="sentences",
        module="fizgig.captioners.joycaption:JoyCaption",
        files=HFRepo("fancyfeast/llama-joycaption-beta-one-hf-llava", "ebf414ea497a020da0f82df3913e5b6cb8e9663a"),
        licence="Llama 3.1 Community License", vram_gb=8.0, size_gb=16.1, instruction=True,
        precisions=(("bf16", 19.0), ("nf4", 8.0)),
        prompt_types=_JC_TYPES, prompt_lengths=_JC_LENGTHS, default_length="long", prompt_extras=_JC_EXTRAS,
        blurb="Detailed training captions; good with people, mirrors and every style",
        hint="An 8B captioner made for training captions: good with people (gender, mirrors and reflections) and "
             "every style. Follows the same instructions as Qwen3-VL. Downloads itself the first time (16 GB); "
             "runs in 4-bit on cards under about 19 GB free."),
    CaptionerDescription(
        key="wd_eva02_large_v3", label="WD EVA02 Large v3 (booru tags)", kind="tags",
        module="fizgig.captioners.wd_tagger:WDTagger",
        files=HFRepo("SmilingWolf/wd-eva02-large-tagger-v3", "b25b82a03f7282e41aa2f257a52c7583b710bd1c"),
        licence="Apache-2.0", vram_gb=2.0, size_gb=1.3, blurb="Booru tags, for anime and illustration",
        download_skip=("model.onnx", "model.msgpack"),
        hint="Booru tags (1girl, long hair, smile...) for anime and illustration: what Anima and anime SDXL "
             "models were trained on. Downloads itself the first time (1.3 GB)."),
)
