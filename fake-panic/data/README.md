# Fake Panic data folder

The data (input/output + embeddings) for each run will be stored in this folder in a json file.

This folder is ignored by git, so if you want to share your data you'll need to do so outside of git/github.

For example, if you want to tar up and compress all the json files in this directory, you could use something like:

```sh
tar -czvf "fake-panic-data-$(date +%Y%m%d%H%M%S).tar.gz" *.json
```
