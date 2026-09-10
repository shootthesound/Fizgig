# Fizgig v5.6.1

A fix for a start-up refusal in 5.6.0.

## Fixed: MiniMax H3 runs with Optimised Likeness Learning off refused to start

With Optimised Likeness Learning unticked (the Style preset, or any full-model run) a 5.6.0 run stopped before its first step with "only 200 of 208 targeted Linears were wrapped". The safety check that counts the model's target layers was counting the text token refiner's eight, which the LoRA no longer trains, so it disagreed with what was actually wrapped and refused. The count now matches the builder exactly. Nothing about training changes; runs with Optimised Likeness on were never affected.
