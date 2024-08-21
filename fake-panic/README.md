# This is Fake Panic

It works like panic: outputs are fed back in as inputs recursively, bouncing
back-and-forth between an image generation model (currently flux schnell) and a
captioning model (currently BLIP). It also calculates the embedding (using
ImageBind) for each output in the sequence.

The `generate_image_embeddings.py` script is the workhorse, and it calls out to
Replicate-hosted models to do the actual work.

Once completed, the sequence of outputs (including embeddings) is stored in
`data/output.json` as an array inside a top-level dict, keyed by the ISO8601
timestamp of the start of the run (so that timestamp serves as a "run ID").

Each element in the inner array represents the output of a single model, and
looks something like this. The `seq_no` is the index of this output in the
particular run, so `seq_no` 0 is the first, etc. You can find the initial promt
for a given run by looking at the `input` key for the `seq_no` 0 output.

```json
{
  "embedding": [
      0.013916094787418842,
      -0.009030221961438656,
      0.035814229398965836,
      -0.04640622064471245,
      0.013718576170504093,
      -0.03190242871642113,
      -0.009067106060683727,
      // ...
      // many more embedding values in here
      // ...
      -0.03326720371842384,
      0.0017629198264330626,
      -0.04796994850039482,
      0.01854240521788597,
      0.007344389334321022,
      0.028173308819532394,
      0.054895900189876556,
      0.01086434442549944
  ],
  "input": "a group of sheep grazing on lush grass",
  "seq_no": 0,
  "type": "image"

```

## Install/use

I use [rye](https://rye.astral.sh) for managing my python environments these
days, and this project was set up with rye.

_However_, all the code I wrote is just in the `generate_embeddings.py` file in
this folder. So just run that with `rye run generate_embeddings.py` and it'll
prompt you for the initial text prompt and you're away.

You'll also need to set a `REPLICATE_API_TOKEN` environment variable. If you're
connected to SOCY, Ben can share one connected to the School's account.

## TODO

- [ ] move the script stuff into the actual `fake_panic:main` function so that
      it can be run as a script
- [ ] provide cli arg for number of runs and initial prompt
- [ ] (maybe) find a better way of serialising the embeddings in the json file
      (base64-encoded binary?)
