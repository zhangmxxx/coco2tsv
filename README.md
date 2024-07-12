# Convert Objects365 to tsv format

See [this page](https://github.com/microsoft/scene_graph_benchmark/tree/main/tools/mini_tsv) for a general knowledge of tsv format.

Objects365 is actually a coco-like dataset, thus pycocotools is used to parse the annotation file.

### What's different

1. Using buffer for tsv file generation. (For large datasets, it's hard to store all the rows in RAM)
2. Multi-threaded converting. (To be implemented)

