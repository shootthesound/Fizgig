# Fizgig v9.3.1: Training maps for several subjects, and Fizgig LoRA Isolation fixes

**Training maps cover every subject.** With Multi Concept or Fizgig LoRA Isolation, each subject now gets at least one of the map pictures, so four subjects get four. An Isolation run also includes one of each subject's side-by-side pictures, the subject beside a stranger: on the Trigger word map, the trigger should land on your subject and stay off the stranger. That's a direct look at whether Isolation is keeping other people themselves. When a caption names more than one subject's trigger, the Trigger word map shows each trigger on its own too. [Training maps guide](https://github.com/shootthesound/Fizgig/blob/master/docs/TRAINING_MAPS.md)

**Fizgig LoRA Isolation: the side-by-side pictures are found again.** Every subject's side-by-side pictures share file names, and they're built in the cache folder rather than your training folder. So with two subjects, their pictures were treated as one, and the Per-image loss window couldn't show them. Each subject's are now kept apart and found. Resuming an Isolation run started before this update starts the side-by-side pictures' per-image history afresh; training itself is unaffected.

**Training maps: a Map opacity slider** makes the map fainter or stronger over the picture. 100% is the usual look; double-click to put it back.

**Updating:** run the updater as usual. Pods you use from your own Fizgig follow your computer's version when they're free.
