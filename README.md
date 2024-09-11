# Embedding Trajectories experiments

This repo is a place for us to experiment (research woo!) with different ideas
about semantic trajectories, genAI, embeddings,
[TDA](https://en.wikipedia.org/wiki/Topological_data_analysis) and more.

## Motivation and research questions

Anyone who's played with Panic (or watched others play with Panic) has probably
had one of these questions cross their mind at some point.

The purpose of this project is to give us answers to these questions which are
both quantifiable and satisfying (i.e. we feel like they represent deeper truths
about the process).

The hope is that these methods are useful outside Panic as well, to understand
and predict semantic trajectories/narrative structure/the shape of information
flows in animal and machine communication).

### how did it get _here_ from _that_ initial prompt?

- was it predictable that it would end up here?
- how sensitive is it to the input, i.e. would it still have ended up here with
  a _slightly_ different prompt?
- how sensitive is it to the random seed(s) of the models?

### is it stuck?

- the text/images it's generating now seem to be "semantically stable"; will it
  ever move on to a different topic?
- is it predictable which initial prompts lead to a "stuck" trajectory?

### has it done this before?

- how similar is this run's trajectory to other runs?
- what determines whether they'll be similar? initial prompt, or something else?

### which models have the biggest impact on what happens?

- do certain models dominate the trajectory? or is it an emergent property of
  the interactions between all models in the network?

## Repo structure

- `fake-panic/` contains a couple of python scripts for running a "fake" version
  of Panic (Panic itself is in a separate repo, ask [Ben](ben.swift@anu.edu.au)
  if you need access).

- `lit-trajectories/` contains python scripts for chunking + embedding classic
  texts from **The Canon**^TM and looking at their trajectories in embedding
  space.

There will be more subfolders as we come up with more experiments---although we
can also share code between experiments too. For now, just get stuff done and we
can refactor as we go.

See the READMEs in each subfolder for more info about each project.

## TODO

- create a unified data model for all trajectories (timeseries of semantic data
  points), perhaps with Pydantic?
- move the TDA code across from Sungyeon's other TDA repo
- switch to a [vector database](https://github.com/asg017/sqlite-vec) for the
  embeddings (in fact, for all data)
- run this on cybersonic

## Authors

Ben Swift, Sungyeon Hong, hopefully others :) Check the commit history to see
who did what.

## Licence

MIT
