"""The Start tab's News card: newest first. Each item is a date, a title, a body (plain text, paragraphs split on a
blank line) and optional links ({"text", "href"}). The card shows the newest few; the full articles are docs/NEWS.md."""

_ARTICLE = "https://github.com/shootthesound/Fizgig/blob/master/docs/NEWS.md"

NEWS = [
    {
        "date": "25 Oct 2026",
        "title": "Masks from words with SAM 3.1",
        "body": (
            "Make masks of anything by naming it: on the Image Prep tab's masks card, choose SAM 3.1 and type what "
            "the masks show (person, hair, the dog) and what to leave out (glasses). Meta's Segment Anything finds "
            "it, with much sharper edges than the person finder.\n\n"
            "Download it once from the card (1.75 GB), preview it on two random pictures, then make masks."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#25-oct-2026-masks-from-words-with-sam-31"},
                  {"text": "The guide", "href": "https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md"}],
    },
    {
        "date": "25 Oct 2026",
        "title": "Masked training",
        "body": (
            "Train on the part of each picture that matters: your subject, not the room around it. A "
            "black-and-white mask beside a picture says which part counts; the picture still trains whole, so "
            "the LoRA sees your subject in their scene but stops copying the scene. " + 'In our tests of character LoRAs, masks significantly raised both the best likeness a run reached, by eye and in likeness scores, and how quickly it got there, on every recent model. It seems especially effective on MOSS-Image with its default Clean-end share, Slide.'
            + "\n\nTick it on the Training tab (Masked training card), and make masks for a whole folder on the "
            "Image Prep tab, or paint your own. Since 8.9.7 it also has prior preservation outside the mask, so "
            "backgrounds and the model's look stay the model's own."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#25-oct-2026-masked-training"},
                  {"text": "The guide", "href": "https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md"}],
    },
    {
        "date": "24 Oct 2026",
        "title": "Faster training on 40 GB+ cards",
        "body": (
            "On cards with 40 GB or more, which is most pods (A100, A40, A6000, L40S, H100, RTX PRO 6000), Auto now "
            "turns gradient checkpointing off when the run fits with room to spare: 13-27 % faster per step on "
            "Krea 2, Qwen Image 2.1, MOSS-Image, Z-Image and MiniMax H3. Where it doesn't all fit, the last blocks "
            "go without it: 2-21 % faster.\n\n"
            "Previews keep at least 4 GB free either way, and cards of 32 GB and under train as before. Restart "
            "your pods to get it."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#24-oct-2026-faster-training-on-40-gb-cards"}],
    },
    {
        "date": "22 Oct 2026",
        "title": "Your other GPUs, at home or in the cloud, speeding up your run",
        "body": (
            "A helper is another computer running Fizgig: a second PC or laptop, a second GPU in this computer, or a "
            "RunPod or Vast.ai pod. Add one on the Helpers tab, and it renders your previews so training never stops "
            "for them, takes over a run while you need your computer (Move to... on the run card), or trains a share "
            "of each epoch alongside your GPU.\n\n"
            "In a share, each GPU trains its own copy of the LoRA on its part of the images, and the copies' changes "
            "are added into one LoRA at every combine: in testing its likeness matched a one-GPU run. On a fast "
            "network, combine every epoch for the closest result. It pays most on long runs. Training on a pod? Your "
            "home GPU can help it, and its checkpoints come home as they're saved."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#22-oct-2026-your-other-gpus-at-home-or-in-the-cloud-speeding-up-your-run"},
                  {"text": "Watch the video", "href": "https://www.youtube.com/watch?v=7Ao0gE2b2qw"}],
    },
    {
        "date": "20 Oct 2026",
        "title": "MOSS-Image likeness sooner, and Base previews with the SFT look",
        "body": (
            "A new Clean-end share default for MOSS-Image, Slide, brings likeness in early and keeps it: on our most "
            "difficult dataset it was the best of eight settings we tried.\n\n"
            "Give Base previews the SFT look (Samples tab) gives your Base previews much of SFT's look with no model "
            "swap. We still recommend leaving the Turbo LoRA off for previews. And update ComfyUI: many people's "
            "Fizgig LoRAs work better since its latest update."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#20-oct-2026-moss-image-likeness-sooner-and-base-previews-with-the-sft-look"},
                  {"text": "Watch the video", "href": "https://www.youtube.com/watch?v=UCoCtM8yrBk"}],
    },
    {
        "date": "20 Oct 2026",
        "title": "MOSS-Image Turbo previews: fast, but don't judge likeness on them",
        "body": (
            "The MOSS-Image Turbo LoRAs aren't mature yet: they can reduce likeness a lot in previews. Use them to "
            "check a run is on track quickly, then judge likeness on full-quality previews: set Turbo LoRA strength to "
            "0 on the Samples tab (Steps and CFG go back with it). The gallery's Turbo line now says this too.\n\n"
            "New for every model: How often to preview and How often to save. Preview every 5 epochs, say, then from "
            "epoch 10 every epoch."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#20-oct-2026-moss-image-turbo-previews-fast-but-dont-judge-likeness-on-them"}],
    },
    {
        "date": "20 Oct 2026",
        "title": "Kromoss out as a preview model",
        "body": (
            "Kromoss is no longer offered as the MOSS-Image preview model: it's too different from MOSS-Image Base for "
            "reliable previews of a LoRA trained on Base, and as a base to train on it overcooks. If you set Kromoss on "
            "the preview model or Base DiT row, press Download models for me in Preferences: it points both back to "
            "SFT and MOSS-Image Base.\n\n"
            "The 8-step Turbo LoRA stays, for fast previews of your Base: untick Use the SFT model for samples and set "
            "Turbo LoRA strength to 1 (12 steps)."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#20-oct-2026-kromoss-out-as-a-preview-model-the-turbo-lora-stays"}],
    },
    {
        "date": "19 Oct 2026",
        "title": "MOSS-Image training, fixed",
        "body": (
            "MOSS-Image LoRAs now learn likeness much sooner, and the picture holds its composition from epoch to "
            "epoch. MOSS's own schedule put only about 11% of training steps where faces and detail are learned; "
            "Fizgig's new Clean-end share puts 30% there. By eye the pictures are far sharper and more accurate, with "
            "no body horror. It's the default: there's nothing to set."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#19-oct-2026-moss-image-training-fixed"}],
    },
]
